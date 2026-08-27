import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Stores experience tuple.
        state, next_state: (hidden_dim) tensor
        action: int
        reward: float
        done: bool
        """
        self.buffer.append((state.detach(), action, reward, next_state.detach(), done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.stack(state)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, dtype=torch.float32)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class SpatialGlimpseExtractor(nn.Module):
    """
    Given a batch of high-resolution frames (B, C, H, W), extracts a specific 
    spatial glimpse crop based on discrete action index, reducing pixel processing by 75%.
    Vectorized using F.grid_sample for massive speedups at K=32.
    """
    def __init__(self):
        super().__init__()
        # Precompute affine matrices for the 6 glimpse regions.
        # Affine matrix format: [[sx, 0, tx], [0, sy, ty]]
        # Coordinates in grid_sample are [-1, 1].
        
        # 0: Full Frame -> scale 1.0 (but we want to downsample to H/2, W/2, which grid_sample handles via output size)
        full = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        # 1: TL -> scale 0.5 (zoom in 2x, so sx=0.5), tx=-0.5, ty=-0.5
        tl = [[0.5, 0.0, -0.5], [0.0, 0.5, -0.5]]
        # 2: TR
        tr = [[0.5, 0.0, 0.5], [0.0, 0.5, -0.5]]
        # 3: BL
        bl = [[0.5, 0.0, -0.5], [0.0, 0.5, 0.5]]
        # 4: BR
        br = [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        # 5: Center
        center = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
        
        self.matrices = torch.tensor([full, tl, tr, bl, br, center], dtype=torch.float32)
        
    def forward(self, frames, glimpse_indices):
        """
        frames: (B, 3, H, W)
        glimpse_indices: 1D tensor of B ints in [0, 5]
        """
        B, C, H, W = frames.shape
        crop_h, crop_w = H // 2, W // 2
        
        device = frames.device
        self.matrices = self.matrices.to(device)
        
        # Ensure glimpse_indices is a tensor and properly formatted (Fixes Issue 2)
        if not isinstance(glimpse_indices, torch.Tensor):
            glimpse_indices = torch.tensor(glimpse_indices, dtype=torch.long, device=device)
            
        glimpse_indices = glimpse_indices.view(-1)
        
        # Gather the specific affine matrix for each item in the batch
        theta = self.matrices[glimpse_indices] # (B, 2, 3)
        
        # Generate the grid and sample
        grid = F.affine_grid(theta, size=(B, C, crop_h, crop_w), align_corners=False)
        cropped_frames = F.grid_sample(frames, grid, align_corners=False)
        
        return cropped_frames

class WorkerAgent(nn.Module):
    """
    Phase 2 Temporal-Spatial Local Agent.
    Extends AdaFrame-RL by integrating Spatial Glimpse cropping.
    """
    def __init__(self, num_classes=101, feature_dim=512, hidden_dim=256, frame_hops=[1, 2, 4, 8]):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.frame_hops = frame_hops
        self.num_temporal_actions = len(frame_hops)
        self.num_spatial_actions = 6 # {Full, TL, TR, BL, BR, Center}
        self.num_joint_actions = self.num_temporal_actions * self.num_spatial_actions
        
        # Spatial Cropper
        self.glimpse_extractor = SpatialGlimpseExtractor()
        
        # 1. Feature backbone
        weights = models.MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(1280, feature_dim)
        
        # 2. Temporal GRU cell
        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=hidden_dim)
        
        # 3. Action Recognition Classifier Head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 4. Joint DQN Policy Head (Predicts 24 Q-Values)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_joint_actions)
        )
        
    def extract_feature(self, frame_tensor, glimpse_indices):
        cropped = self.glimpse_extractor(frame_tensor, glimpse_indices)
        feats = self.features(cropped)
        feats = self.pool(feats).view(cropped.size(0), -1)
        return F.relu(self.projection(feats))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def forward(self, frame_tensor, h_prev, glimpse_indices):
        """
        Executes a single step across a batch of streams simultaneously.
        Inputs:
            frame_tensor: (B, 3, H, W)
            h_prev: (B, hidden_dim)
            glimpse_indices: list/tensor of B spatial actions selected in the PREVIOUS step.
        """
        feat = self.extract_feature(frame_tensor, glimpse_indices)
        h_new = self.gru_cell(feat, h_prev)
        
        logits = self.classifier(h_new)
        q_values = self.policy_net(h_new) # (B, 24)
        
        return h_new, logits, q_values

    def decode_action(self, action_idx):
        """Decodes discrete joint action into temporal hop and spatial glimpse."""
        temporal_idx = action_idx // self.num_spatial_actions
        spatial_idx = action_idx % self.num_spatial_actions
        hop = self.frame_hops[temporal_idx]
        return hop, spatial_idx
