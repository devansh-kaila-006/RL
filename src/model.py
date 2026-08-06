import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torchvision.models as models

class AdaFrameFeatureExtractor(nn.Module):
    """
    Lightweight 2D CNN spatial feature extractor utilizing pre-trained MobileNetV2 architecture.
    Optimized for fast inference on standard CPUs without requiring GPU hardware accelerators.
    """
    def __init__(self, feature_dim=512, use_pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        base_model = models.mobilenet_v2(weights=weights)
        # Extract features without the classification classifier block
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(1280, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (B, C, H, W) -> single video frame slice
        feats = self.features(x)
        feats = self.pool(feats).view(x.size(0), -1)
        emb = self.dropout(self.relu(self.projection(feats)))
        return emb  # -> (B, feature_dim)


class AdaFrameModel(nn.Module):
    """
    Unified AdaFrame-RL network architecture coupling spatial-temporal memory with active RL frame hopping.
    
    Components:
    1. Spatial Extractor: MobileNetV2 embedding frame pixels into latent representation.
    2. Temporal Memory: GRUCell maintaining recurrent visual state h_t across sampled temporal steps.
    3. Action Classifier: Linear layer outputting classification logits for UCF-101 action classes.
    4. RL Policy Network (DQN): Predicts Q-values for temporal frame hop decisions A_t in {1, 2, 4, 8}.
    """
    def __init__(self, num_classes, feature_dim=512, hidden_dim=256, frame_hops=[1, 2, 4, 8]):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.frame_hops = frame_hops
        self.num_actions = len(frame_hops)
        
        # 1. Feature backbone
        self.backbone = AdaFrameFeatureExtractor(feature_dim=feature_dim, use_pretrained=True)
        
        # 2. Temporal GRU cell
        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=hidden_dim)
        
        # 3. Action Recognition Classifier Head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 4. DQN Policy Head for Active Frame Hop Selection
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
    def init_hidden(self, batch_size, device):
        """Returns zeros tensor initialized as initial temporal memory state h_0."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def forward_frame_step(self, frame_tensor, h_prev):
        """
        Executes a single temporal perception step on an inspected video frame.
        Inputs:
            frame_tensor: (B, 3, 224, 224) pixel observation at time t.
            h_prev: (B, hidden_dim) temporal memory state from previous step.
        Outputs:
            h_new: Updated temporal memory state.
            logits: Classification predictions at current observation step.
            q_values: Predicted Q-values across available frame hopping actions {1, 2, 4, 8}.
        """
        feat = self.backbone(frame_tensor)
        h_new = self.gru_cell(feat, h_prev)
        logits = self.classifier(h_new)
        q_values = self.policy_net(h_new)
        return h_new, logits, q_values

    def forward_exhaustive_baseline(self, video_tensor):
        """
        Naive Exhaustive Video Baseline (No Frame Skipping):
        Processes 100% of video frames chronologically without skipping for comparative evaluation.
        Input: video_tensor of shape (B, T, C, H, W).
        Returns: final classification logits and total evaluated frame count.
        """
        batch_size, seq_len, _, _, _ = video_tensor.shape
        device = video_tensor.device
        h_t = self.init_hidden(batch_size, device=device)
        
        for t in range(seq_len):
            frame_t = video_tensor[:, t, :, :, :]
            feat_t = self.backbone(frame_t)
            h_t = self.gru_cell(feat_t, h_t)
            
        logits = self.classifier(h_t)
        return logits, seq_len  # 0% frames skipped
