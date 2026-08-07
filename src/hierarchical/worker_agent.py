import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialGlimpseExtractor(nn.Module):
    """
    Given a batch of high-resolution frames (B, C, H, W), extracts a specific 
    spatial glimpse crop based on discrete action index, reducing pixel processing by 75%.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames, glimpse_idx):
        """
        frames: (B, 3, H, W)
        glimpse_idx: int in [0, 5]
        """
        B, C, H, W = frames.shape
        crop_h, crop_w = H // 2, W // 2
        
        # We process each item in the batch differently if necessary, 
        # but for simplicity we assume the whole batch shares the same glimpse action
        # during independent stream processing (batch size 1 per stream).
        
        if glimpse_idx == 0:
            # Full Frame (Downsampled)
            return F.interpolate(frames, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
        elif glimpse_idx == 1:
            # Top-Left
            return frames[:, :, :crop_h, :crop_w]
        elif glimpse_idx == 2:
            # Top-Right
            return frames[:, :, :crop_h, crop_w:]
        elif glimpse_idx == 3:
            # Bottom-Left
            return frames[:, :, crop_h:, :crop_w]
        elif glimpse_idx == 4:
            # Bottom-Right
            return frames[:, :, crop_h:, crop_w:]
        elif glimpse_idx == 5:
            # Center Crop
            ch_start, cw_start = H // 4, W // 4
            return frames[:, :, ch_start:ch_start+crop_h, cw_start:cw_start+crop_w]
        else:
            raise ValueError(f"Invalid glimpse_idx {glimpse_idx}")

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
        
    def extract_feature(self, frame_tensor, glimpse_idx):
        cropped = self.glimpse_extractor(frame_tensor, glimpse_idx)
        feats = self.features(cropped)
        feats = self.pool(feats).view(cropped.size(0), -1)
        return F.relu(self.projection(feats))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def forward_step(self, frame_tensor, h_prev, glimpse_idx=0):
        """
        Executes a single step.
        Inputs:
            frame_tensor: (B, 3, H, W)
            h_prev: (B, hidden_dim)
            glimpse_idx: The spatial action selected in the PREVIOUS step to crop current frame.
        """
        feat = self.extract_feature(frame_tensor, glimpse_idx)
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
