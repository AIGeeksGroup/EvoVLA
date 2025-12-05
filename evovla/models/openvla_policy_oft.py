"""
OpenVLA-OFT Policy Wrapper

Wraps OpenVLA-OFT model with PPO-compatible components:
- Diagonal Gaussian policy head for continuous action sampling
- Dual value heads for extrinsic/intrinsic returns
- Action normalization/denormalization
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

try:
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None


class OpenVLAPolicyOFT(nn.Module):
    """
    OpenVLA-OFT based policy for PPO training.
    
    Architecture:
    - VLA backbone: Pre-trained OpenVLA-OFT (frozen or fine-tuned)
    - Action head: Linear projection from LLM hidden states to 7-DoF actions
    - Value heads: Dual heads for extrinsic and intrinsic value estimation
    - Log std: Learnable diagonal covariance for Gaussian policy
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        mode: str = "ppo",
        freeze_backbone: bool = True,
        enable_lora: bool = False,
        lora_adapter_path: Optional[str] = None,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        unfreeze_layers: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path: Path to OpenVLA-OFT pretrained model
            device: Computation device
            mode: "ppo" for training, "inference" for evaluation
            freeze_backbone: Whether to freeze VLA backbone
            enable_lora: Enable LoRA adaptation
            lora_adapter_path: Path to existing LoRA adapter
            lora_r: LoRA rank
            lora_alpha: LoRA alpha coefficient
            lora_dropout: LoRA dropout probability
            lora_target_modules: LoRA target modules
            unfreeze_layers: Backbone layer patterns to unfreeze
        """
        super().__init__()
        
        self.model_path = model_path
        self.device = device
        self.mode = mode
        self.freeze_backbone = freeze_backbone
        self.using_lora = False
        self.lora_adapter_path = lora_adapter_path
        self.lora_config_dict = None
        self._trainable_backbone_params: List[nn.Parameter] = []
        self._trainable_backbone_names: List[str] = []
        self._grad_safety_handles: List[Any] = []
        
        # Load VLA model and processor
        print(f"[OpenVLAPolicyOFT] Loading model: {model_path}")
        self._load_vla_model(model_path)
        
        # Setup action scaling
        self._setup_action_scaling()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vla.parameters():
                param.requires_grad = False
            print("  ✓ VLA backbone frozen")
        else:
            print("  ✓ VLA backbone trainable")
        
        # Initialize LoRA if requested
        target_modules = self._parse_target_modules(lora_target_modules)
        if enable_lora or lora_adapter_path:
            self._init_lora(
                adapter_path=lora_adapter_path,
                target_modules=target_modules,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            self._cast_lora_parameters(torch.float32)
            self._install_gradient_safety_hooks()
        
        # Unfreeze specific layers if requested
        if unfreeze_layers:
            patterns = self._parse_target_modules(unfreeze_layers)
            if patterns:
                self._unfreeze_backbone_layers(patterns)
                self._install_gradient_safety_hooks()
        
        # Get LLM hidden dimension
        if hasattr(self.vla, 'llm_backbone'):
            self.llm_dim = self.vla.llm_backbone.config.hidden_size
        else:
            self.llm_dim = 4096  # Llama-2-7B default
        
        print(f"  ✓ llm_dim = {self.llm_dim}")
        
        # Training components (only for PPO mode)
        if mode == "ppo":
            _model_param = next(self.vla.parameters())
            _device = _model_param.device
            _dtype = _model_param.dtype
            
            # Diagonal Gaussian log std
            self.log_std = nn.Parameter(
                torch.zeros(7, device=_device, dtype=_dtype)
            )
            
            # Dual value heads
            self.value_head_ext = nn.Sequential(
                nn.Linear(self.llm_dim, 256, device=_device),
                nn.ReLU(),
                nn.Linear(256, 1, device=_device)
            )
            
            self.value_head_int = nn.Sequential(
                nn.Linear(self.llm_dim, 256, device=_device),
                nn.ReLU(),
                nn.Linear(256, 1, device=_device)
            )
            
            print(f"  ✓ Training components added: log_std, value_heads")
            print(f"  ✓ Device aligned: device={_device}, dtype={_dtype}")
        
        print(f"✓ OpenVLAPolicyOFT initialized (mode={mode})")
    
    def _load_vla_model(self, model_path: str):
        """Load VLA model and processor using official OpenVLA utilities."""
        try:
            # Try to import OpenVLA utilities
            from experiments.robot.openvla_utils import get_vla, get_processor
            
            class Config:
                pretrained_checkpoint = model_path
                load_in_8bit = False
                load_in_4bit = False
                center_crop = True
                unnorm_key = "libero_spatial_no_noops"
                model_family = "openvla"
            
            self.cfg = Config()
            self.vla = get_vla(self.cfg)
            self.processor = get_processor(self.cfg)
            
        except ImportError:
            # Fallback to transformers loading
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.vla = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            class Config:
                unnorm_key = "libero_spatial_no_noops"
            self.cfg = Config()
    
    def _parse_target_modules(self, modules) -> Optional[List[str]]:
        """Parse target modules from string or list."""
        if modules is None:
            return None
        if isinstance(modules, str):
            return [m.strip() for m in modules.split(',') if m.strip()]
        return [str(m).strip() for m in modules if str(m).strip()]
    
    def _setup_action_scaling(self):
        """Initialize action normalization statistics."""
        self.action_low = None
        self.action_high = None
        self.action_mask = None
        self.action_mask_inverted = None
        self.task_name = None
        self.action_low_original = None
        self.action_high_original = None
    
    def _init_lora(self, adapter_path=None, target_modules=None, r=32, alpha=16, dropout=0.05):
        """Initialize LoRA adaptation."""
        try:
            from peft import LoraConfig, get_peft_model, PeftModel
        except ImportError as exc:
            raise ImportError("LoRA requires peft library: pip install peft") from exc
        
        default_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        modules = target_modules or default_targets
        
        adapter_path = Path(adapter_path) if adapter_path else None
        if adapter_path and adapter_path.exists():
            print(f"  ✓ Loading LoRA adapter from {adapter_path}")
            self.vla = PeftModel.from_pretrained(self.vla, str(adapter_path), is_trainable=True)
            self.lora_adapter_path = str(adapter_path)
            self.lora_config_dict = {
                name: cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
                for name, cfg in self.vla.peft_config.items()
            }
        else:
            print(f"  ✓ Enabling LoRA (r={r}, alpha={alpha}, dropout={dropout}, target={modules})")
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=modules,
            )
            self.vla = get_peft_model(self.vla, lora_config)
            self.lora_config_dict = {"default": lora_config.to_dict()}
        
        self.using_lora = True
    
    def _unfreeze_backbone_layers(self, patterns: List[str]):
        """Unfreeze backbone layers matching given patterns."""
        if not patterns:
            return
        
        lowered = [p.lower() for p in patterns]
        matched = 0
        for name, param in self.vla.named_parameters():
            lower_name = name.lower()
            if any(pat in lower_name for pat in lowered):
                if not param.requires_grad:
                    param.requires_grad = True
                if id(param) not in {id(p) for p in self._trainable_backbone_params}:
                    self._trainable_backbone_params.append(param)
                    self._trainable_backbone_names.append(name)
                    matched += 1
        
        if matched > 0:
            print(f"  ✓ Unfroze {matched} backbone parameters matching pattern={patterns}")
        else:
            print(f"  ⚠️ No backbone parameters found matching pattern={patterns}")
    
    def _cast_lora_parameters(self, dtype=torch.float32):
        """Cast LoRA parameters to specified dtype."""
        if not self.using_lora:
            return
        
        import peft
        
        cast_count = 0
        for module in self.vla.modules():
            if hasattr(module, 'lora_A') and isinstance(module, peft.tuners.lora.LoraLayer):
                for adapter_name, lora_A in module.lora_A.items():
                    if lora_A.weight.dtype != dtype:
                        module.lora_A[adapter_name].weight = nn.Parameter(lora_A.weight.float().to(dtype))
                        cast_count += 1
                for adapter_name, lora_B in module.lora_B.items():
                    if lora_B.weight.dtype != dtype:
                        module.lora_B[adapter_name].weight = nn.Parameter(lora_B.weight.float().to(dtype))
                        cast_count += 1
        
        if cast_count:
            print(f"  ✓ LoRA weights cast to {dtype} ({cast_count} matrices)")
    
    def _install_gradient_safety_hooks(self):
        """Install hooks to sanitize gradients (prevent NaN/Inf)."""
        if self._grad_safety_handles:
            return
        
        try:
            import peft
        except ImportError:
            return
        
        def sanitize_in_place(tensor):
            if tensor is None:
                return None
            return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        
        def make_hook():
            def hook(module, grad_input, grad_output):
                if grad_output:
                    for g in grad_output:
                        if g is not None:
                            g.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                if grad_input:
                    sanitized = tuple(sanitize_in_place(g) for g in grad_input)
                    return sanitized
                return grad_input
            return hook
        
        for module in self.vla.modules():
            if isinstance(module, peft.tuners.lora.LoraLayer):
                handle = module.register_full_backward_hook(make_hook())
                self._grad_safety_handles.append(handle)
    
    def set_task_norm_stats(self, task_name: str):
        """
        Set task-specific normalization statistics.
        
        Args:
            task_name: Task name ('stack', 'jujube_cup', 'bridge')
        """
        import json
        
        self.task_name = task_name
        
        stats_file = Path(__file__).parent.parent.parent / 'discoverse_norm_stats.json'
        
        if not stats_file.exists():
            print(f"⚠️ Norm stats file not found: {stats_file}")
            print("  Please run: python scripts/compute_discoverse_norm_stats.py")
            return
        
        with open(stats_file, 'r') as f:
            all_stats = json.load(f)
        
        if task_name not in all_stats:
            print(f"⚠️ No norm stats for task {task_name}")
            return
        
        action_stats = all_stats[task_name]['action']
        
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.action_low = torch.tensor(action_stats["q01"], dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_stats["q99"], dtype=torch.float32, device=device)
        mask_list = action_stats.get("mask", [True] * len(action_stats["q01"]))
        self.action_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
        self.action_mask_inverted = ~self.action_mask
        self.action_low_original = self.action_low.clone()
        self.action_high_original = self.action_high.clone()
        
        print(f"✅ Loaded norm_stats for {task_name}")
        print(f"  q01: {action_stats['q01']}")
        print(f"  q99: {action_stats['q99']}")
    
    def update_action_bounds(self, new_low, new_high, keep_original=False):
        """Update action clipping bounds (for dynamic expansion during training)."""
        device = self.action_low.device if self.action_low is not None else (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        
        new_low_t = torch.as_tensor(new_low, dtype=torch.float32, device=device)
        new_high_t = torch.as_tensor(new_high, dtype=torch.float32, device=device)
        
        if keep_original or self.action_low_original is None:
            self.action_low_original = new_low_t.clone()
        if keep_original or self.action_high_original is None:
            self.action_high_original = new_high_t.clone()
        
        self.action_low = new_low_t
        self.action_high = new_high_t
    
    def get_action_bounds(self) -> Optional[Dict]:
        """Get current action bounds."""
        if self.action_low is None or self.action_high is None:
            return None
        return {
            'low': self.action_low.detach().cpu().tolist(),
            'high': self.action_high.detach().cpu().tolist(),
            'original_low': self.action_low_original.detach().cpu().tolist() if self.action_low_original is not None else None,
            'original_high': self.action_high_original.detach().cpu().tolist() if self.action_high_original is not None else None,
            'mask': self.action_mask.detach().cpu().tolist() if self.action_mask is not None else None,
        }
    
    def _normalize_pre_activation(self, pre_action: torch.Tensor) -> torch.Tensor:
        """Clip linear layer output to action space bounds."""
        if self.action_low is None or self.action_high is None or self.action_mask is None:
            return pre_action.float()
        
        result = pre_action.float().clone()
        
        if torch.any(self.action_mask):
            result[self.action_mask] = torch.max(
                torch.min(result[self.action_mask], self.action_high[self.action_mask]),
                self.action_low[self.action_mask]
            )
        
        if self.action_mask_inverted is not None and torch.any(self.action_mask_inverted):
            result[self.action_mask_inverted] = torch.clamp(result[self.action_mask_inverted], 0.0, 1.0)
        
        return result
    
    def decode_from_pre_activation(self, pre_action: torch.Tensor) -> torch.Tensor:
        """Convert linear layer output to executable environment action."""
        if not isinstance(pre_action, torch.Tensor):
            device = self.action_low.device if self.action_low is not None else (
                torch.device(self.device) if isinstance(self.device, str) else self.device
            )
            pre_action = torch.tensor(pre_action, dtype=torch.float32, device=device)
        
        return self._normalize_pre_activation(pre_action)
    
    def get_sft_trainable_parameter_groups(self) -> List[Dict]:
        """Get parameter groups for SFT training."""
        groups = []
        
        if hasattr(self, 'action_head_sft'):
            head_params = [p for p in self.action_head_sft.parameters() if p.requires_grad]
            if head_params:
                groups.append({'name': 'action_head', 'params': head_params})
        
        if self.using_lora:
            lora_params = [
                p for name, p in self.vla.named_parameters()
                if p.requires_grad and ('lora_' in name or 'lora' in name)
            ]
            if lora_params:
                groups.append({'name': 'lora', 'params': self._unique_params(lora_params)})
        
        if self._trainable_backbone_params:
            groups.append({
                'name': 'backbone_partial',
                'params': self._unique_params(self._trainable_backbone_params)
            })
        
        return groups
    
    def get_sft_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters for SFT."""
        params = []
        for group in self.get_sft_trainable_parameter_groups():
            params.extend(group['params'])
        return self._unique_params(params)
    
    def _unique_params(self, params):
        """Return unique parameters by id."""
        unique = []
        seen = set()
        for param in params:
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                unique.append(param)
        return unique
    
    def sample_action(self, mu: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from diagonal Gaussian distribution.
        
        Args:
            mu: Mean [batch, 7] or [7,]
            log_std: Log standard deviation [7,]
        
        Returns:
            action: Sampled action [batch, 7]
            log_prob: Log probability [batch,]
        """
        std = torch.exp(log_std)
        
        # Reparameterization trick
        eps = torch.randn_like(mu)
        action = mu + std * eps
        
        # Compute log_prob (diagonal Gaussian)
        log_prob = -0.5 * (
            ((action - mu) / std) ** 2 +
            2 * torch.log(std) +
            np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob
    
    def forward_train(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass (differentiable version).
        
        Args:
            obs: Dict with 'image' (PIL Image), 'task_description' (str)
        
        Returns:
            mu: Action mean [7,] with gradients
            log_std: Log standard deviation [7,]
            v_ext: Extrinsic value scalar
            v_int: Intrinsic value scalar
        """
        # Construct prompt and process input
        prompt = f"In: What action should the robot take to {obs['task_description'].lower()}?\nOut:"
        inputs = self.processor(prompt, obs['image']).to(self.device, dtype=torch.bfloat16)
        
        # Create dummy labels for forward pass
        dummy_labels = torch.full_like(
            inputs['input_ids'],
            fill_value=-100,
            dtype=torch.long,
            device=self.device
        )
        
        requires_backbone_grad = (
            not self.freeze_backbone
            or self.using_lora
            or bool(self._trainable_backbone_params)
        )
        
        vla_kwargs = dict(
            labels=dummy_labels,
            output_hidden_states=True,
            use_cache=False,
        )
        
        if requires_backbone_grad:
            self.vla.eval()
            outputs = self.vla(**inputs, **vla_kwargs)
        else:
            self.vla.eval()
            with torch.no_grad():
                outputs = self.vla(**inputs, **vla_kwargs)
        
        # Extract last layer hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
            vla_features = last_hidden[:, -1, :]  # [1, llm_dim]
        else:
            vla_features = outputs.last_hidden_state[:, -1, :]
        
        vla_features = vla_features.float()  # [1, llm_dim], keep gradients
        
        # Create action head if not exists (lazy initialization)
        if not hasattr(self, 'action_head_sft'):
            self.action_head_sft = nn.Linear(self.llm_dim, 7).to(
                device=vla_features.device,
                dtype=vla_features.dtype
            )
            nn.init.normal_(self.action_head_sft.weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.action_head_sft.bias)
            print(f"  [SFT] Created action_head_sft: {self.llm_dim} -> 7")
        
        # Predict action (with gradients!)
        mu = self.action_head_sft(vla_features).squeeze(0)  # [7,]
        
        # Compute values
        v_ext = self.value_head_ext(vla_features).squeeze()
        v_int = self.value_head_int(vla_features).squeeze()
        
        return mu, self.log_std, v_ext, v_int
    
    def predict_action_deterministic(self, image: Image.Image, task_description: str) -> np.ndarray:
        """
        Deterministic action prediction (inference).
        
        Args:
            image: PIL Image
            task_description: Task description string
        
        Returns:
            action: numpy array [7,] (denormalized)
        """
        obs = {'image': image, 'task_description': task_description}
        mu, _, _, _ = self.forward_train(obs)
        action = self.decode_from_pre_activation(mu)
        return action.detach().cpu().numpy()
    
    def save_for_inference(self, save_path: str):
        """
        Save checkpoint for inference.
        
        Saves VLA weights + action_head_sft (if exists).
        Does NOT save training components (log_std, value_heads).
        """
        if self.using_lora and hasattr(self.vla, 'base_model') and hasattr(self.vla.base_model, 'model'):
            base_state_iterator = self.vla.base_model.model.state_dict().items()
        else:
            base_state_iterator = self.vla.state_dict().items()
        
        base_state_dict = {k: v.detach().cpu() for k, v in base_state_iterator}
        
        checkpoint = {
            'vla_state_dict': base_state_dict,
            'norm_stats': self.vla.norm_stats if hasattr(self.vla, 'norm_stats') else None,
            'config': {
                'model_path': self.model_path,
                'unnorm_key': self.cfg.unnorm_key,
                'llm_dim': self.llm_dim
            },
        }
        
        bounds = self.get_action_bounds()
        if bounds:
            checkpoint['action_bounds'] = bounds
        
        if hasattr(self, 'action_head_sft'):
            checkpoint['action_head_sft_state_dict'] = self.action_head_sft.state_dict()
        
        if self.using_lora:
            lora_state = None
            if get_peft_model_state_dict is not None:
                lora_state = get_peft_model_state_dict(self.vla)
            
            if lora_state:
                checkpoint['lora_state_dict'] = {k: v.detach().cpu() for k, v in lora_state.items()}
            
            checkpoint['lora_config'] = self.lora_config_dict
            if self.lora_adapter_path:
                checkpoint['lora_adapter_path'] = self.lora_adapter_path
            checkpoint['using_lora'] = True
        
        torch.save(checkpoint, save_path)
        
        if hasattr(self, 'action_head_sft'):
            print(f"✓ Inference checkpoint saved: {save_path}")
            print(f"  Contains: VLA backbone + SFT Action Head")
        else:
            print(f"✓ Inference checkpoint saved: {save_path}")
    
    def load_for_inference(self, checkpoint_path: str):
        """Load inference checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        base_state = checkpoint['vla_state_dict']
        
        # Detect LoRA format
        use_lora = bool(checkpoint.get('using_lora'))
        if not use_lora:
            use_lora = any('lora' in key for key in base_state.keys())
        
        if use_lora:
            self._ensure_lora_initialized(
                config_dict=checkpoint.get('lora_config'),
                adapter_path=checkpoint.get('lora_adapter_path'),
            )
            
            target_model = (
                self.vla.base_model.model
                if hasattr(self.vla, 'base_model') and hasattr(self.vla.base_model, 'model')
                else self.vla
            )
            target_model.load_state_dict(base_state, strict=False)
            
            lora_state_dict = checkpoint.get('lora_state_dict')
            if lora_state_dict and set_peft_model_state_dict is not None:
                set_peft_model_state_dict(self.vla, lora_state_dict, adapter_name='default', ignore_mismatched_sizes=False)
                print("  ✓ LoRA weights restored from checkpoint")
            
            self.using_lora = True
            if checkpoint.get('lora_config'):
                self.lora_config_dict = checkpoint['lora_config']
        else:
            self.vla.load_state_dict(base_state, strict=True)
        
        if checkpoint.get('norm_stats'):
            self.vla.norm_stats = checkpoint['norm_stats']
        
        # Load action head if present
        if checkpoint.get('action_head_sft_state_dict'):
            if hasattr(self, 'action_head_sft'):
                self.action_head_sft.load_state_dict(checkpoint['action_head_sft_state_dict'])
                print(f"✓ Loaded inference checkpoint: {checkpoint_path}")
                print(f"  Contains SFT-trained action_head")
            else:
                print(f"⚠️ Checkpoint contains action_head_sft, but model not initialized")
                print(f"  Please call forward_train() first to create action_head_sft")
        else:
            print(f"✓ Loaded inference checkpoint: {checkpoint_path}")
        
        # Load action bounds
        bounds = checkpoint.get('action_bounds')
        if bounds and bounds.get('low') is not None and bounds.get('high') is not None:
            self.update_action_bounds(bounds['low'], bounds['high'], keep_original=False)
            device = self.action_low.device
            if bounds.get('original_low') is not None and bounds.get('original_high') is not None:
                self.action_low_original = torch.tensor(bounds['original_low'], dtype=torch.float32, device=device)
                self.action_high_original = torch.tensor(bounds['original_high'], dtype=torch.float32, device=device)
            if bounds.get('mask') is not None:
                self.action_mask = torch.tensor(bounds['mask'], dtype=torch.bool, device=device)
                self.action_mask_inverted = ~self.action_mask
    
    def _ensure_lora_initialized(self, config_dict=None, adapter_path=None):
        """Ensure LoRA is initialized."""
        if self.using_lora:
            return
        
        normalized = self._normalize_lora_config_dict(config_dict)
        target_modules = normalized.get('target_modules') if normalized else None
        r = normalized.get('r', normalized.get('lora_r', 32)) if normalized else 32
        alpha = normalized.get('lora_alpha', normalized.get('alpha', 16)) if normalized else 16
        dropout = normalized.get('lora_dropout', 0.0) if normalized else 0.0
        self._init_lora(
            adapter_path=adapter_path,
            target_modules=target_modules,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
    
    def _normalize_lora_config_dict(self, config_dict):
        """Normalize LoRA config dictionary."""
        if config_dict is None:
            return None
        if isinstance(config_dict, dict):
            if 'peft_type' in config_dict:
                return config_dict
            for value in config_dict.values():
                if isinstance(value, dict) and 'peft_type' in value:
                    return value
        return None


if __name__ == "__main__":
    # Test
    print("=" * 80)
    print("Testing OpenVLAPolicyOFT")
    print("=" * 80)
    
    model_path = "/path/to/openvla-7b-oft-libero"
    
    try:
        policy = OpenVLAPolicyOFT(model_path, device="cuda", mode="ppo", freeze_backbone=True)
        
        # Test inference
        print("\n[Test 1] Inference...")
        test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        test_task = "pick up the green block"
        
        action = policy.predict_action_deterministic(test_image, test_task)
        print(f"✓ Action shape: {action.shape}")
        
        # Test training forward
        print("\n[Test 2] Training forward...")
        obs = {'image': test_image, 'task_description': test_task}
        mu, log_std, v_ext, v_int = policy.forward_train(obs)
        print(f"✓ mu shape: {mu.shape}")
        print(f"✓ v_ext: {v_ext.item():.4f}")
        print(f"✓ v_int: {v_int.item():.4f}")
        
        # Test sampling
        print("\n[Test 3] Action sampling...")
        action_sampled, log_prob = policy.sample_action(mu, log_std)
        print(f"✓ Sampled action shape: {action_sampled.shape}")
        print(f"✓ Log prob: {log_prob.item():.4f}")
        
        print("\n" + "=" * 80)
        print("🎉 All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

