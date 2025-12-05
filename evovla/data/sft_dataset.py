"""
SFT Dataset for DISCOVERSE Demonstrations

Loads DISCOVERSE demonstration data (obs_action.json + mp4 videos)
and converts to OpenVLA-OFT SFT training format.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    print("Warning: av not installed, video loading disabled")


class SFTDataset(Dataset):
    """
    DISCOVERSE Demonstration Dataset.
    
    Reads DISCOVERSE generated demonstration data (obs_action.json + mp4 videos)
    and converts to OpenVLA-OFT SFT training format.
    """
    
    def __init__(
        self,
        data_root: str,
        task_name: str,
        task_description: str
    ):
        """
        Args:
            data_root: DISCOVERSE data root directory
            task_name: Task name (e.g., 'block_bridge_place')
            task_description: Task description for prompt
        """
        self.data_root = Path(data_root)
        self.task_name = task_name
        self.task_description = task_description
        
        # Scan all successful episodes
        self.episodes: List[Dict] = []
        task_dir = self.data_root / task_name
        
        if not task_dir.exists():
            raise ValueError(f"Task directory not found: {task_dir}")
        
        # Collect all episode directories
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir():
                json_file = ep_dir / 'obs_action.json'
                cam0_file = ep_dir / 'cam_0.mp4'
                cam1_file = ep_dir / 'cam_1.mp4'
                
                if json_file.exists() and cam0_file.exists():
                    self.episodes.append({
                        'dir': ep_dir,
                        'json': json_file,
                        'cam0': cam0_file,
                        'cam1': cam1_file if cam1_file.exists() else cam0_file
                    })
        
        print(f"[SFTDataset] Loaded task: {task_name}")
        print(f"  Found {len(self.episodes)} successful episodes")
        
        # Build global index: (episode_idx, frame_idx)
        self.samples: List[tuple] = []
        for ep_idx, ep_info in enumerate(self.episodes):
            with open(ep_info['json'], 'r') as f:
                data = json.load(f)
            
            num_frames = len(data['time'])
            for frame_idx in range(num_frames):
                self.samples.append((ep_idx, frame_idx))
        
        print(f"  Total samples: {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.
        
        Returns:
            dict: {
                'full_image': PIL.Image,
                'wrist_image': PIL.Image,
                'state': np.array([7,]),
                'action': np.array([7,]),
                'prev_action': np.array([7,]),
                'task_description': str
            }
        """
        ep_idx, frame_idx = self.samples[idx]
        ep_info = self.episodes[ep_idx]
        
        # Load obs_action.json
        with open(ep_info['json'], 'r') as f:
            data = json.load(f)
        
        # Extract state and action
        state = np.array(data['obs']['jq'][frame_idx], dtype=np.float32)
        action = np.array(data['act'][frame_idx], dtype=np.float32)
        
        # Get previous action for smoothing
        prev_idx = max(frame_idx - 1, 0)
        prev_action = np.array(data['act'][prev_idx], dtype=np.float32)
        
        # Load images from video
        full_image = self._extract_frame(ep_info['cam0'], frame_idx)
        wrist_image = self._extract_frame(ep_info['cam1'], frame_idx)
        
        return {
            'full_image': full_image,
            'wrist_image': wrist_image,
            'state': state,
            'action': action,
            'prev_action': prev_action,
            'task_description': self.task_description
        }
    
    def _extract_frame(self, video_path: Path, frame_idx: int) -> Image.Image:
        """Extract specified frame from mp4."""
        if not AV_AVAILABLE:
            return Image.new('RGB', (448, 448), color=(128, 128, 128))
        
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = 'NONKEY'
            
            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_idx:
                    img_array = frame.to_ndarray(format='rgb24')
                    pil_img = Image.fromarray(img_array)
                    container.close()
                    return pil_img
            
            container.close()
        except Exception as e:
            print(f"Failed to extract frame: {e}")
        
        return Image.new('RGB', (448, 448), color=(128, 128, 128))


if __name__ == "__main__":
    # Test dataset
    data_root = "/path/to/discoverse/data"
    
    if Path(data_root).exists():
        print("Testing SFTDataset...")
        
        try:
            dataset = SFTDataset(
                data_root=data_root,
                task_name='stack_block',
                task_description='stack the green block on the blue block'
            )
            
            print(f"\nLoading first sample...")
            sample = dataset[0]
            print(f"  full_image: {sample['full_image'].size}")
            print(f"  state shape: {sample['state'].shape}")
            print(f"  action shape: {sample['action'].shape}")
            print(f"  task: {sample['task_description']}")
            
            print(f"\n✅ SFTDataset test passed!")
            
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print(f"Data directory not found: {data_root}")

