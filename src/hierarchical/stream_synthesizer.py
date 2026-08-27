import torch
import numpy as np

class MultiStreamSynthesizer:
    """
    Simulates a dense, concurrent multi-camera CCTV video surveillance grid by
    stacking K independent video sequences into a unified persistent environment.
    
    When a video stream reaches its end, it automatically respawns with a new
    random video sequence from the dataset to mimic continuous 24/7 feeds.
    """
    def __init__(self, dataloader_iterator, K_streams=4, max_sequence_length=150):
        self.iterator = dataloader_iterator
        self.K = K_streams
        self.max_len = max_sequence_length
        
        self.streams = [None] * self.K
        self.stream_labels = [None] * self.K
        self.stream_pointers = [0] * self.K
        self.stream_lengths = [0] * self.K
        self.prev_frames = [None] * self.K
        
        self._initialize_streams()
        
    def _initialize_streams(self):
        """Populate initial K streams from the asynchronous queue."""
        for k in range(self.K):
            self._respawn_stream(k)
            
    def _respawn_stream(self, k):
        """Fetch a pre-decoded video from the background thread queue and load into stream k."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            raise RuntimeError("Dataloader Iterator ran out of data! Ensure it is wrapped in an infinite loop generator.")
            
        # Handle tuple unpacking robustly (Fixes Issue 4)
        if len(batch) == 2:
            video, label = batch
        elif len(batch) >= 3:
            video, label = batch[0], batch[1]
        else:
            video = batch[0]
            label = 0
            
        # The DataLoader returns a batched tensor (1, T, C, H, W) since we will configure it with batch_size=1
        video = video.squeeze(0) # (T, C, H, W)
        label = int(label.squeeze(0).item()) if isinstance(label, torch.Tensor) else label
        
        # Ensure it is bounded
        seq_len = min(video.shape[0], self.max_len)
        self.streams[k] = video[:seq_len] # (T, C, H, W)
        self.stream_labels[k] = label
        self.stream_pointers[k] = 0
        self.stream_lengths[k] = seq_len
        self.prev_frames[k] = None
        
    def get_current_frames(self):
        """
        Retrieves the current frame at the active pointer for all K streams.
        Returns:
            frames: (K, C, H, W) Tensor
            labels: list of K integer labels
            entropies: (K,) Tensor simulating visual motion complexity 
        """
        frames = []
        labels = []
        entropies = []
        
        for k in range(self.K):
            t = self.stream_pointers[k]
            frame = self.streams[k][t] # (C, H, W)
            frames.append(frame)
            labels.append(self.stream_labels[k])
            
            # Motion entropy: Mean absolute difference from previous frame (Fixes Issue 11)
            if self.prev_frames[k] is None:
                entropy = 0.0
            else:
                entropy = torch.mean(torch.abs(frame.float() - self.prev_frames[k].float())).item()
                
            self.prev_frames[k] = frame
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
