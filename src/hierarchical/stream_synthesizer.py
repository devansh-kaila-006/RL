import torch
import numpy as np

class MultiStreamSynthesizer:
    """
    Simulates a dense, concurrent multi-camera CCTV video surveillance grid by
    stacking K independent video sequences into a unified persistent environment.
    
    When a video stream reaches its end, it automatically respawns with a new
    random video sequence from the dataset to mimic continuous 24/7 feeds.
    """
    def __init__(self, dataset, K_streams=4, max_sequence_length=150):
        self.dataset = dataset
        self.K = K_streams
        self.max_len = max_sequence_length
        
        self.streams = [None] * self.K
        self.stream_labels = [None] * self.K
        self.stream_pointers = [0] * self.K
        self.stream_lengths = [0] * self.K
        self.dataset_idx = 0
        
        self._initialize_streams()
        
    def _initialize_streams(self):
        """Populate initial K streams from dataset."""
        for k in range(self.K):
            self._respawn_stream(k)
            
    def _respawn_stream(self, k):
        """Fetch a new video from dataset and load into stream k."""
        idx = np.random.randint(0, len(self.dataset))
        video, label, _ = self.dataset[idx]
        
        # Ensure it is bounded
        seq_len = min(video.shape[0], self.max_len)
        self.streams[k] = video[:seq_len] # (T, C, H, W)
        self.stream_labels[k] = label
        self.stream_pointers[k] = 0
        self.stream_lengths[k] = seq_len
        
    def get_current_frames(self):
        """
        Retrieves the current frame at the active pointer for all K streams.
        Returns:
            frames: (K, C, H, W) Tensor
            labels: list of K integer labels
            entropies: (K,) Tensor simulating visual motion complexity 
                       (calculated roughly as spatial variance for budget purposes).
        """
        frames = []
        labels = []
        entropies = []
        
        for k in range(self.K):
            t = self.stream_pointers[k]
            frame = self.streams[k][t] # (C, H, W)
            frames.append(frame)
            labels.append(self.stream_labels[k])
            
            # Simulated visual entropy: Standard deviation of pixel intensities
            # In a real deployed system, this would be optical flow or consecutive frame difference.
            entropy = frame.float().std().item()
            entropies.append(entropy)
            
        return torch.stack(frames), labels, torch.tensor(entropies)
        
    def step_streams(self, hop_actions):
        """
        Advances each stream k by its requested hop stride.
        hop_actions: List or tensor of K integers (e.g., 1, 2, 4, 8)
        If a stream exceeds its duration, it automatically respawns.
        """
        for k in range(self.K):
            self.stream_pointers[k] += int(hop_actions[k])
            
            # Check for end of stream
            if self.stream_pointers[k] >= self.stream_lengths[k]:
                self._respawn_stream(k)
                
    def reset(self):
        """Hard reset of all streams."""
        self._initialize_streams()
