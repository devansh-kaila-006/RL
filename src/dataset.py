import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UCF101Dataset(Dataset):
    """
    PyTorch Dataset Loader for UCF-101 Action Recognition.
    Enforces strict train/test partitioning using official split annotations in data/splits/.
    Supports dynamic video frame decoding via OpenCV and graceful synthetic fallbacks during offline prototyping.
    """
    def __init__(self, data_dir="data", split="train01", seq_len=32, resize=(224, 224), subset_classes=None, allow_synthetic_fallback=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        self.resize = resize
        self.allow_synthetic_fallback = allow_synthetic_fallback
        
        self.splits_dir = os.path.join(data_dir, "splits")
        self.video_dir = os.path.join(data_dir, "UCF-101")
        
        # 1. Parse class indicator dictionary
        self.class_to_idx, self.idx_to_class = self._load_class_map()
        
        # Optionally filter to a specific subset of classes for rapid prototyping on CPU
        if subset_classes is not None:
            self.subset_ids = {self.class_to_idx[cls_name] for cls_name in subset_classes if cls_name in self.class_to_idx}
        else:
            self.subset_ids = set(self.class_to_idx.values())
            
        # 2. Parse split file lists (trainlist01.txt vs testlist01.txt)
        self.video_items = self._load_split_items()
        
    def _load_class_map(self):
        class_map = {}
        idx_map = {}
        class_file = os.path.join(self.splits_dir, "classInd.txt")
        if os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        idx = int(parts[0]) - 1  # Convert 1-indexed to 0-indexed PyTorch class labels
                        cls_name = parts[1]
                        class_map[cls_name] = idx
                        idx_map[idx] = cls_name
        else:
            # Fallback demo mapping if classInd.txt is unreadable
            demo_classes = ["Basketball", "Pushups", "BenchPress", "SoccerJumping", "PlayingGuitar"]
            for idx, name in enumerate(demo_classes):
                class_map[name] = idx
                idx_map[idx] = name
        return class_map, idx_map
        
    def _load_split_items(self):
        items = []
        if "train" in self.split:
            list_file = os.path.join(self.splits_dir, f"trainlist{self.split[-2:]}.txt")
        else:
            list_file = os.path.join(self.splits_dir, f"testlist{self.split[-2:]}.txt")
            
        if os.path.exists(list_file):
            with open(list_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "train" in self.split:
                        path_str, cls_idx_str = line.split()
                        cls_idx = int(cls_idx_str) - 1
                    else:
                        path_str = line
                        cls_name = path_str.split("/")[0]
                        cls_idx = self.class_to_idx.get(cls_name, 0)
                        
                    if cls_idx in self.subset_ids:
                        items.append((path_str, cls_idx))
        else:
            print(f"Warning: Split file {list_file} not found. Using synthetic test clips for verification.")
            for i in range(10):
                cls_idx = list(self.subset_ids)[i % len(self.subset_ids)]
                items.append((f"{self.idx_to_class[cls_idx]}/v_synth_clip_{i:02d}.avi", cls_idx))
                
        return items

    def _decode_video(self, video_path):
        """Decodes up to seq_len frames from an AVI video clip using OpenCV."""
        if not os.path.exists(video_path):
            if self.allow_synthetic_fallback:
                # Return synthetic motion tensors if manual video download is in progress
                frames = np.random.randint(0, 256, (self.seq_len, self.resize[1], self.resize[0], 3), dtype=np.uint8)
                return self._preprocess_frames(frames)
            else:
                raise FileNotFoundError(f"Video file not found at {video_path}")
                
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.resize)
            frames.append(frame_resized)
        cap.release()
        
        if not frames:
            frames = [np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8)]
            
        frames = np.array(frames)
        total_frames = len(frames)
        
        # Sample uniformly across the timeline to achieve exact seq_len frames
        indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
        sampled_frames = frames[indices]
        
        return self._preprocess_frames(sampled_frames)
        
    def _preprocess_frames(self, frames):
        """Convert array of shape (T, H, W, C) to PyTorch tensor (T, C, H, W) normalized to [0, 1]."""
        tensor_frames = torch.from_numpy(frames).float() / 255.0
        tensor_frames = tensor_frames.permute(0, 3, 1, 2)  # -> (T, C, H, W)
        
        # ImageNet normalization statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_frames = (tensor_frames - mean) / std
        return tensor_frames

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, index):
        rel_path, cls_idx = self.video_items[index]
        full_path = os.path.join(self.video_dir, rel_path.replace("/", os.sep))
        video_tensor = self._decode_video(full_path)
        return video_tensor, cls_idx, rel_path


def get_ucf_loaders(data_dir="data", batch_size=4, seq_len=32, subset_classes=None, allow_synthetic=True, num_workers=0):
    """
    Helper function returning standard PyTorch DataLoaders for train and test splits.
    """
    train_dataset = UCF101Dataset(data_dir=data_dir, split="train01", seq_len=seq_len, 
                                  subset_classes=subset_classes, allow_synthetic_fallback=allow_synthetic)
    test_dataset = UCF101Dataset(data_dir=data_dir, split="test01", seq_len=seq_len, 
                                 subset_classes=subset_classes, allow_synthetic_fallback=allow_synthetic)
                                 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset.idx_to_class
