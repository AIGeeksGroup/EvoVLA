"""
Stage Aligned Reward (SAR) Module

Uses triplet contrastive learning to evaluate stage completion and generate
dense reward signals. Implements Equations 2-4 from the paper.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import json
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional


class StageAlignedReward(nn.Module):
    """
    Stage Aligned Reward (SAR)
    
    Uses a frozen CLIP model to compute image-text similarity through triplet
    contrastive learning, evaluating stage completion with temporal smoothing
    to produce stable reward signals.
    
    Paper: Equations 2-4 implementation
    """
    
    def __init__(
        self,
        triplets_path: str,
        device: str = "cuda",
        tau: float = 0.05,
        alpha: float = 0.05,
        clip_model: str = "ViT-B/32"
    ):
        """
        Args:
            triplets_path: Path to triplets JSON file
            device: Computation device
            tau: Temperature parameter (paper: 0.05)
            alpha: Temporal smoothing coefficient (paper: 0.05)
            clip_model: CLIP model variant to use
        """
        super().__init__()
        
        self.device = device
        self.tau = tau
        self.alpha = alpha
        
        # Load pre-generated triplets
        with open(triplets_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.triplets = data['triplets']
        self.num_stages = data['num_stages']
        
        # Load frozen CLIP model
        print(f"Loading CLIP model for SAR...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=device)
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Pre-encode all text descriptions
        print(f"Pre-encoding text descriptions for {self.num_stages} stages...")
        self.text_embeddings: Dict[int, Dict[str, torch.Tensor]] = {}
        self._precompute_text_embeddings()
        
        # Temporal smoothing history
        self.u_bar: Dict[int, float] = {}
        
        print(f"✓ SAR initialized ({self.num_stages} stages)")
    
    def _truncate_text(self, text: str, max_length: int = 76) -> str:
        """Truncate text to fit CLIP's context_length limit."""
        if len(text) > max_length:
            return text[:max_length]
        return text
    
    def _precompute_text_embeddings(self):
        """Pre-encode all text descriptions to avoid redundant computation."""
        with torch.no_grad():
            for stage_key, triplet in self.triplets.items():
                stage_idx = int(stage_key.split('_')[1])
                
                # Truncate long texts
                texts = [
                    self._truncate_text(triplet['positive']),
                    self._truncate_text(triplet['negative']),
                    self._truncate_text(triplet['hard_negative'])
                ]
                
                # Tokenize texts
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                
                # Encode to vectors
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)
                
                self.text_embeddings[stage_idx] = {
                    'positive': text_features[0],
                    'negative': text_features[1],
                    'hard_negative': text_features[2]
                }
    
    @torch.no_grad()
    def compute(
        self,
        obs_image: np.ndarray,
        current_stage: int
    ) -> Tuple[float, float]:
        """
        Compute stage-aligned reward.
        
        Args:
            obs_image: Observation image (H, W, 3) numpy array or PIL Image
            current_stage: Current stage index
        
        Returns:
            r_stage: Stage reward (float)
            u_k: Current stage score (float, for hallucination rate computation)
        """
        # Preprocess image
        if isinstance(obs_image, np.ndarray):
            obs_image = Image.fromarray(obs_image.astype(np.uint8))
        
        image_input = self.clip_preprocess(obs_image).unsqueeze(0).to(self.device)
        
        # Encode image
        image_features = self.clip_model.encode_image(image_input)
        image_features = F.normalize(image_features, dim=-1)
        
        # Get text embeddings for this stage
        if current_stage not in self.text_embeddings:
            return 0.0, 0.0
        
        text_emb = self.text_embeddings[current_stage]
        
        # Compute similarities (Equation 2)
        s_pos = (image_features @ text_emb['positive']).item()
        s_neg = (image_features @ text_emb['negative']).item()
        s_hard = (image_features @ text_emb['hard_negative']).item()
        
        # Contrastive score: u_k = sigmoid(tau * (s+ - max(s-, sh-)))
        contrastive_score = s_pos - max(s_neg, s_hard)
        u_k = torch.sigmoid(torch.tensor(self.tau * contrastive_score)).item()
        
        # Temporal smoothing (Equation 3)
        if current_stage not in self.u_bar:
            self.u_bar[current_stage] = u_k
        else:
            self.u_bar[current_stage] = (
                (1 - self.alpha) * self.u_bar[current_stage] +
                self.alpha * u_k
            )
        
        # Compute reward (Equation 4)
        r_stage = u_k - self.u_bar[current_stage]
        
        return r_stage, u_k
    
    def reset(self):
        """Reset smoothing history (call at episode start)."""
        self.u_bar = {}
    
    def get_stage_score(self, obs_image: np.ndarray, stage_idx: int) -> float:
        """Get score for specified stage (for evaluation)."""
        _, u_k = self.compute(obs_image, stage_idx)
        return u_k


if __name__ == "__main__":
    # Test SAR module
    print("Testing SAR module...")
    
    triplets_path = "stage_dictionaries/stack_block_triplets.json"
    
    if Path(triplets_path).exists():
        sar = StageAlignedReward(triplets_path, device="cuda")
        
        # Create test image
        test_image = Image.new('RGB', (448, 448), color=(100, 100, 100))
        test_array = np.array(test_image)
        
        # Test computation
        r_stage, u_k = sar.compute(test_array, current_stage=0)
        print(f"\nTest results:")
        print(f"  Stage 0 reward: {r_stage:.4f}")
        print(f"  Stage 0 score:  {u_k:.4f}")
        
        print(f"\n✓ SAR module test passed!")
    else:
        print(f"Triplets file not found: {triplets_path}")

