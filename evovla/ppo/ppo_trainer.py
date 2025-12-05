"""
PPO Trainer (Continuous Gaussian Policy)

Features:
1. Rollout collection with parallel environments
2. SAR+POE reward computation and integration (Equation 1)
3. GAE advantage estimation
4. PPO clipped objective update
5. Checkpoint management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional

from evovla.utils.torch_utils import compute_gae


class PPOTrainer:
    """
    PPO Trainer (Paper implementation)
    
    Core features:
    - Rollout collection (with SAR+POE reward integration)
    - GAE computation (gamma=0.995, lambda=0.95)
    - PPO clipped objective update
    - Dual value head training
    """
    
    def __init__(self, policy, sar, poe, config: Dict):
        """
        Args:
            policy: OpenVLAPolicyOFT instance
            sar: StageAlignedReward instance
            poe: PoseBasedExploration instance
            config: Configuration dictionary or OmegaConf object
        """
        self.policy = policy
        self.sar = sar
        self.poe = poe
        self.cfg = config
        
        # Extract config parameters
        ppo_cfg = config.get('ppo', config) if hasattr(config, 'get') else config.ppo
        
        # Optimizers (separate for policy and world model)
        policy_params = [p for p in policy.parameters() if p.requires_grad]
        self.policy_optimizer = AdamW(
            policy_params,
            lr=ppo_cfg.get('learning_rate', 3e-4)
        )
        
        self.world_model_optimizer = AdamW(
            poe.parameters(),
            lr=ppo_cfg.get('learning_rate', 3e-4)
        )
        
        # PPO hyperparameters (Paper Section 3.2)
        self.gamma = ppo_cfg.get('gamma', 0.995)
        self.lambda_gae = ppo_cfg.get('lambda_gae', 0.95)
        self.clip_range = ppo_cfg.get('clip_range', 0.2)
        self.entropy_coef = ppo_cfg.get('entropy_coef', 0.01)
        self.value_coef = ppo_cfg.get('value_coef', 0.5)
        self.max_grad_norm = ppo_cfg.get('max_grad_norm', 0.5)
        
        # Reward weights (Equation 1)
        rewards_cfg = config.get('rewards', config) if hasattr(config, 'get') else config.rewards
        self.rho = rewards_cfg.get('rho', 0.6)
        
        print(f"✓ PPOTrainer initialized")
        print(f"  PPO params: gamma={self.gamma}, lambda={self.lambda_gae}, clip={self.clip_range}")
        print(f"  Reward params: rho={self.rho}")
    
    def collect_rollout_single_env(self, env, num_steps: int = 2048) -> Dict:
        """
        Collect rollout from single environment.
        
        Args:
            env: DISCOVERSE environment instance
            num_steps: Rollout length
        
        Returns:
            rollout_data: Dictionary containing trajectory data
        """
        rollout_data = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'values_ext': [],
            'values_int': [],
            'rewards_ext': [],
            'rewards_int': [],
            'rewards_total': [],
            'dones': [],
            'stage_scores': []
        }
        
        obs = env.reset()
        
        for step in range(num_steps):
            # Policy forward (training mode)
            mu, log_std, v_ext, v_int = self.policy.forward_train(obs)
            
            # Sample action (pre-activation)
            action_pre, log_prob = self.policy.sample_action(mu, log_std)
            
            # Decode to executable action
            action_env = self.policy.decode_from_pre_activation(action_pre)
            
            # Convert to numpy for execution
            action_np = action_pre.detach().cpu().numpy()
            action_env_np = action_env.detach().cpu().numpy()
            
            # Execute action
            next_obs, r_ext, done, info = env.step(action_env_np)
            
            # Compute intrinsic rewards
            # 1. SAR reward (Equations 2-4)
            current_stage = env.current_stage
            sar_image = obs.get('full_image')
            if sar_image is None:
                base_image = obs['image']
                if isinstance(base_image, Image.Image):
                    sar_image = np.array(base_image)
                else:
                    sar_image = base_image
            
            r_stage, u_k = self.sar.compute(sar_image, current_stage)
            
            # Stage gating mechanism (Paper Appendix)
            advanced, old_stage, new_stage = env.update_stage_gating(r_stage, u_k)
            if advanced:
                self.sar.reset()  # Reset smoothing history on stage transition
            
            # 2. POE reward (Equations 5-6)
            if 'ee_pose' in obs and 'obj_pose' in obs:
                z_t = self.poe.encode_pose(obs['ee_pose'], obs['obj_pose'])
                z_t1 = self.poe.encode_pose(next_obs['ee_pose'], next_obs['obj_pose'])
                
                z_t_tensor = torch.tensor(z_t, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                z_t1_tensor = torch.tensor(z_t1, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
                action_tensor = action_env.unsqueeze(0)
                
                r_cur, r_prog = self.poe.compute_rewards(z_t_tensor, action_tensor, z_t1_tensor)
                r_cur_val = r_cur.item()
                r_prog_val = r_prog.item() if isinstance(r_prog, torch.Tensor) else r_prog
            else:
                r_cur_val = 0.0
                r_prog_val = 0.0
            
            # Total intrinsic reward
            r_int = r_stage + r_cur_val + r_prog_val
            
            # Combined reward (Equation 1)
            r_total = r_ext + self.rho * r_int
            
            # Store data
            rollout_data['observations'].append(obs)
            rollout_data['actions'].append(action_np)
            rollout_data['log_probs'].append(log_prob.item())
            rollout_data['values_ext'].append(v_ext.item())
            rollout_data['values_int'].append(v_int.item())
            rollout_data['rewards_ext'].append(r_ext)
            rollout_data['rewards_int'].append(r_int)
            rollout_data['rewards_total'].append(r_total)
            rollout_data['dones'].append(done)
            rollout_data['stage_scores'].append(u_k)
            
            # Update observation
            obs = next_obs
            
            # Episode end
            if done:
                obs = env.reset()
                self.sar.reset()
        
        return rollout_data
    
    def compute_advantages(self, rollout_data: Dict):
        """
        Compute GAE advantages (Paper GAE(lambda) implementation).
        
        Returns:
            advantages: Combined advantages
            returns_ext: Extrinsic returns
            returns_int: Intrinsic returns
        """
        # Convert to tensors
        rewards_ext = torch.tensor(rollout_data['rewards_ext'], dtype=torch.float32)
        rewards_int = torch.tensor(rollout_data['rewards_int'], dtype=torch.float32)
        values_ext = torch.tensor(rollout_data['values_ext'], dtype=torch.float32)
        values_int = torch.tensor(rollout_data['values_int'], dtype=torch.float32)
        dones = torch.tensor(rollout_data['dones'], dtype=torch.float32)
        
        # Bootstrap value (simplified: assume 0 at end)
        values_ext_bootstrap = torch.cat([values_ext, torch.zeros(1)])
        values_int_bootstrap = torch.cat([values_int, torch.zeros(1)])
        
        # GAE for extrinsic
        advantages_ext, returns_ext = compute_gae(
            rewards_ext.unsqueeze(-1),
            values_ext_bootstrap.unsqueeze(-1),
            dones.unsqueeze(-1),
            gamma=self.gamma,
            lambda_=self.lambda_gae
        )
        
        # GAE for intrinsic
        advantages_int, returns_int = compute_gae(
            rewards_int.unsqueeze(-1),
            values_int_bootstrap.unsqueeze(-1),
            dones.unsqueeze(-1),
            gamma=self.gamma,
            lambda_=self.lambda_gae
        )
        
        # Combined advantages (paper fusion method)
        advantages = (1 - self.rho) * advantages_ext.squeeze() + self.rho * advantages_int.squeeze()
        
        return advantages, returns_ext.squeeze(), returns_int.squeeze()
    
    def update_policy(
        self,
        rollout_data: Dict,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64
    ) -> Dict[str, float]:
        """
        PPO policy update (Paper PPO implementation).
        
        Args:
            rollout_data: Rollout data
            ppo_epochs: Number of PPO update epochs
            mini_batch_size: Mini-batch size
        
        Returns:
            train_stats: Training statistics
        """
        # Compute advantages
        advantages, returns_ext, returns_int = self.compute_advantages(rollout_data)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare data
        observations = rollout_data['observations']
        actions = torch.tensor(np.array(rollout_data['actions']), dtype=torch.float32)
        old_log_probs = torch.tensor(rollout_data['log_probs'], dtype=torch.float32)
        
        num_samples = len(observations)
        
        # Statistics
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        approx_kls = []
        explained_vars_ext = []
        explained_vars_int = []
        
        # PPO update (multiple epochs)
        for epoch in range(ppo_epochs):
            # Mini-batch random sampling
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batch data
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns_ext = returns_ext[batch_indices]
                batch_returns_int = returns_int[batch_indices]
                
                # Re-compute log_prob and value
                batch_log_probs = []
                batch_values_ext = []
                batch_values_int = []
                
                model_device = next(self.policy.parameters()).device
                
                for idx, obs in enumerate(batch_obs):
                    mu, log_std, v_ext, v_int = self.policy.forward_train(obs)
                    
                    action = batch_actions[idx]
                    if not isinstance(action, torch.Tensor):
                        action = torch.tensor(action, dtype=mu.dtype, device=model_device)
                    else:
                        action = action.to(device=model_device, dtype=mu.dtype)
                    
                    # Gaussian log_prob
                    std = torch.exp(log_std).to(mu.dtype)
                    log_prob = -0.5 * (
                        ((action - mu) / std) ** 2 +
                        2 * torch.log(std) +
                        np.log(2 * np.pi)
                    )
                    log_prob = log_prob.sum()
                    
                    batch_log_probs.append(log_prob)
                    batch_values_ext.append(v_ext)
                    batch_values_int.append(v_int)
                
                batch_log_probs = torch.stack(batch_log_probs)
                batch_values_ext = torch.stack(batch_values_ext).squeeze()
                batch_values_int = torch.stack(batch_values_int).squeeze()
                
                # Move tensors to device
                batch_old_log_probs = batch_old_log_probs.to(model_device)
                batch_advantages = batch_advantages.to(model_device)
                batch_returns_ext = batch_returns_ext.to(model_device)
                batch_returns_int = batch_returns_int.to(model_device)
                
                # Importance sampling ratio
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                
                # PPO clipped loss (Paper objective)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss_ext = F.mse_loss(batch_values_ext, batch_returns_ext)
                value_loss_int = F.mse_loss(batch_values_int, batch_returns_int)
                value_loss = value_loss_ext + value_loss_int
                
                # Entropy bonus (Gaussian entropy)
                std = torch.exp(log_std)
                entropy = 0.5 * torch.log(2 * np.pi * np.e * std**2).sum()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Record statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                
                # Policy update quality metrics
                clip_mask = ((ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)).float()
                clip_fractions.append(clip_mask.mean().item())
                
                approx_kl = (batch_old_log_probs - batch_log_probs).mean().item()
                approx_kls.append(approx_kl)
                
                # Value function quality
                explained_var_ext = 1 - torch.var(batch_returns_ext - batch_values_ext) / (torch.var(batch_returns_ext) + 1e-8)
                explained_var_int = 1 - torch.var(batch_returns_int - batch_values_int) / (torch.var(batch_returns_int) + 1e-8)
                explained_vars_ext.append(explained_var_ext.item())
                explained_vars_int.append(explained_var_int.item())
        
        return {
            # Base losses
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            
            # Reward breakdown (Equation 1)
            'mean_reward_ext': np.mean(rollout_data['rewards_ext']),
            'mean_reward_int': np.mean(rollout_data['rewards_int']),
            'mean_reward_total': np.mean(rollout_data['rewards_total']),
            'std_reward_total': np.std(rollout_data['rewards_total']),
            
            # Policy update quality
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': np.mean(approx_kls),
            
            # Value function quality
            'explained_variance_ext': np.mean(explained_vars_ext),
            'explained_variance_int': np.mean(explained_vars_int),
        }
    
    def train(
        self,
        env,
        total_steps: int,
        rollout_length: int = 2048,
        ppo_epochs: int = 10,
        save_interval: int = 25000,
        eval_interval: int = 10000,
        log_interval: int = 100,
        save_dir: str = 'checkpoints/',
        logger=None
    ):
        """
        Complete training loop.
        
        Args:
            env: DISCOVERSE environment
            total_steps: Total training steps
            rollout_length: Rollout length per iteration
            ppo_epochs: PPO update epochs
            save_interval: Checkpoint save interval
            eval_interval: Evaluation interval
            log_interval: Logging interval
            save_dir: Checkpoint save directory
            logger: Logger (optional)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_iterations = total_steps // rollout_length
        global_step = 0
        
        print(f"\nStarting PPO training...")
        print(f"  Total steps: {total_steps}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Rollout length: {rollout_length}")
        
        for iteration in range(num_iterations):
            # Collect rollout
            print(f"\n[Iteration {iteration+1}/{num_iterations}] Collecting rollout...")
            rollout_data = self.collect_rollout_single_env(env, num_steps=rollout_length)
            
            # PPO update
            print(f"  PPO update...")
            train_stats = self.update_policy(rollout_data, ppo_epochs=ppo_epochs)
            
            global_step += rollout_length
            
            # Logging (paper standard, all key metrics)
            if iteration % log_interval == 0:
                log_str = (
                    f"Step {global_step:7d} | "
                    f"Loss: π={train_stats['policy_loss']:.4f} V={train_stats['value_loss']:.4f} | "
                    f"Reward: ext={train_stats['mean_reward_ext']:.3f} int={train_stats['mean_reward_int']:.3f} "
                    f"total={train_stats['mean_reward_total']:.3f}±{train_stats['std_reward_total']:.3f} | "
                    f"H={train_stats['entropy']:.4f} clip={train_stats['clip_fraction']:.2%} KL={train_stats['approx_kl']:.6f} | "
                    f"ExpVar: ext={train_stats['explained_variance_ext']:.2f} int={train_stats['explained_variance_int']:.2f}"
                )
                print(f"  {log_str}")
                
                if logger:
                    logger.info(log_str)
            
            # Save checkpoint
            if (iteration + 1) % (save_interval // rollout_length) == 0:
                checkpoint_path = save_dir / f"checkpoint_{global_step}.pt"
                self.save_checkpoint(checkpoint_path, global_step, train_stats)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Save final model (for inference)
        final_path = save_dir / "final_inference.pt"
        self.policy.save_for_inference(final_path)
        print(f"\n✓ Training complete! Final model: {final_path}")
    
    def save_checkpoint(self, save_path: str, global_step: int, train_stats: Optional[Dict] = None):
        """Save complete checkpoint (with training state)."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'world_optimizer_state_dict': self.world_model_optimizer.state_dict(),
            'global_step': global_step,
            'config': {
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'clip_range': self.clip_range,
                'rho': self.rho
            }
        }
        
        if train_stats:
            checkpoint['train_stats'] = train_stats
        
        torch.save(checkpoint, save_path)

