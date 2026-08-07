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
        
        self._initialize_streams()
        
    def _initialize_streams(self):
        """Populate initial K streams from the asynchronous queue."""
        for k in range(self.K):
            self._respawn_stream(k)
            
    def _respawn_stream(self, k):
        """Fetch a pre-decoded video from the background thread queue and load into stream k."""
        try:
            video, label, _ = next(self.iterator)
        except StopIteration:
            # Should not happen if the dataloader is wrapped in an infinite generator,
            # but if it does, we just reset the iterator (handled externally or raise error).
            raise RuntimeError("Dataloader Iterator ran out of data! Ensure it is wrapped in an infinite loop generator.")
            
        # The DataLoader returns a batched tensor (1, T, C, H, W) since we will configure it with batch_size=1
        video = video.squeeze(0) # (T, C, H, W)
        label = int(label.squeeze(0).item()) if isinstance(label, torch.Tensor) else label
        
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
