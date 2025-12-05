#!/usr/bin/env python3
"""
PPO Training Script for EvoVLA

Features:
1. Load OpenVLA-OFT policy
2. Initialize SAR+POE rewards
3. Create DISCOVERSE environment
4. PPO training loop
5. Checkpoint and logging management
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evovla.models.openvla_policy_oft import OpenVLAPolicyOFT
from evovla.rewards.stage_aligned_reward import StageAlignedReward
from evovla.rewards.pose_based_exploration import PoseBasedExploration
from evovla.ppo.ppo_trainer import PPOTrainer
from evovla.data.discoverse_env import DiscoverseEnv
from evovla.utils.config_utils import load_config
from evovla.utils.torch_utils import set_seed
from evovla.utils.logging_utils import setup_logger


def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('train_ppo', output_dir / 'logs' / 'train.log')
    
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Total steps: {args.total_steps}")
    logger.info(f"  Rollout length: {args.rollout_length}")
    logger.info(f"  Mini-batch size: {args.mini_batch_size}")
    logger.info(f"  PPO epochs: {args.ppo_epochs}")
    logger.info(f"  Freeze backbone: {args.freeze_backbone}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # 1. Load OpenVLA-OFT policy
    logger.info("=" * 60)
    logger.info("Loading OpenVLA-OFT policy...")
    policy = OpenVLAPolicyOFT(
        model_path=args.model_path,
        device="cuda",
        mode="ppo",
        freeze_backbone=args.freeze_backbone
    )
    
    # Set task-specific norm stats
    policy.set_task_norm_stats(args.task)
    logger.info(f"✓ Policy loaded (freeze_backbone={args.freeze_backbone})")
    
    # Load SFT warm-up checkpoint if provided
    if args.sft_checkpoint:
        logger.info("=" * 60)
        logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        
        from PIL import Image
        ckpt = torch.load(args.sft_checkpoint, map_location='cpu')
        has_action_head = 'action_head_sft_state_dict' in ckpt
        
        if has_action_head:
            logger.info("✓ Checkpoint contains action_head_sft, initializing...")
            dummy_image = Image.new('RGB', (448, 448), color=(128, 128, 128))
            dummy_obs = {'image': dummy_image, 'task_description': "dummy task"}
            policy.eval()
            _ = policy.forward_train(dummy_obs)
            logger.info("✓ action_head_sft initialized")
        
        policy.load_for_inference(args.sft_checkpoint)
        logger.info("✓ SFT checkpoint loaded")
    
    # 2. Initialize reward modules
    logger.info("=" * 60)
    logger.info("Initializing SAR and POE...")
    
    triplets_path = args.triplets or f'stage_dictionaries/{args.task}_triplets.json'
    sar = StageAlignedReward(triplets_path, device="cuda")
    logger.info(f"✓ SAR initialized: {sar.num_stages} stages")
    
    poe = PoseBasedExploration(device="cuda")
    logger.info(f"✓ POE initialized")
    
    # 3. Create DISCOVERSE environment
    logger.info("=" * 60)
    logger.info("Creating DISCOVERSE environment...")
    env = DiscoverseEnv(
        args.task,
        headless=True,
        randomize=not args.no_domain_randomization,
        stage_window_size=args.stage_window,
        stage_completion_threshold=args.stage_threshold,
        stage_metric=args.stage_metric,
    )
    logger.info(f"✓ Environment created: {args.task}")
    
    # 4. Create PPO trainer
    logger.info("=" * 60)
    logger.info("Creating PPO trainer...")
    trainer = PPOTrainer(policy, sar, poe, config)
    logger.info(f"✓ Trainer created")
    
    # 5. Training loop
    logger.info("=" * 60)
    logger.info("Starting PPO training...")
    logger.info(f"  Total steps: {args.total_steps}")
    logger.info(f"  Rollout length: {args.rollout_length}")
    logger.info(f"  PPO epochs: {args.ppo_epochs}")
    
    num_iterations = args.total_steps // args.rollout_length
    global_step = 0
    
    for iteration in range(num_iterations):
        iter_start_time = time.time()
        
        # Collect rollout
        logger.info(f"\n[Iteration {iteration+1}/{num_iterations}] Collecting rollout...")
        rollout_start = time.time()
        rollout_data = trainer.collect_rollout_single_env(
            env,
            num_steps=args.rollout_length
        )
        rollout_time = time.time() - rollout_start
        
        logger.info(f"  ✓ Rollout complete: {args.rollout_length} steps, {rollout_time/60:.2f} min")
        logger.info(f"    Avg reward: ext={np.mean(rollout_data['rewards_ext']):.3f}, "
                   f"int={np.mean(rollout_data['rewards_int']):.3f}, "
                   f"total={np.mean(rollout_data['rewards_total']):.3f}")
        
        # PPO update
        logger.info(f"  PPO update...")
        ppo_start = time.time()
        train_stats = trainer.update_policy(
            rollout_data,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size
        )
        ppo_time = time.time() - ppo_start
        iter_time = time.time() - iter_start_time
        
        global_step += args.rollout_length
        
        # Logging
        logger.info(f"  ✓ PPO update complete, {ppo_time/60:.2f} min")
        logger.info(f"  ━━━━ Training Metrics (Step {global_step}) ━━━━")
        logger.info(f"  Policy Loss:  {train_stats['policy_loss']:.6f}")
        logger.info(f"  Value Loss:   {train_stats['value_loss']:.4f}")
        logger.info(f"  Entropy:      {train_stats['entropy']:.4f}")
        logger.info(f"  Reward (ext): {train_stats['mean_reward_ext']:.4f}")
        logger.info(f"  Reward (int): {train_stats['mean_reward_int']:.4f}")
        logger.info(f"  Reward (total): {train_stats['mean_reward_total']:.4f}")
        logger.info(f"  ━━━━ Performance ━━━━")
        logger.info(f"  Rollout time:  {rollout_time/60:.2f} min")
        logger.info(f"  PPO time:      {ppo_time/60:.2f} min")
        logger.info(f"  Iteration time: {iter_time/60:.2f} min")
        logger.info(f"  Steps/sec:     {args.rollout_length/rollout_time:.1f}")
        
        # Save checkpoint
        if (iteration + 1) % (args.save_interval // args.rollout_length) == 0:
            checkpoint_path = output_dir / 'checkpoints' / f"checkpoint_{global_step}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path, global_step, train_stats)
            logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_inference.pt"
    policy.save_for_inference(final_path)
    logger.info(f"\n✓ Training complete! Final model: {final_path}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EvoVLA PPO Training')
    
    # Task parameters
    parser.add_argument('--task', type=str, required=True,
                       choices=['bridge', 'jujube_cup', 'stack'],
                       help='Task name')
    parser.add_argument('--model_path', type=str,
                       default='/path/to/openvla-7b-oft',
                       help='OpenVLA-OFT model path')
    parser.add_argument('--triplets', type=str, default=None,
                       help='Triplets file path')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    
    # Training parameters
    parser.add_argument('--config', type=str,
                       default='configs/base_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--total_steps', type=int, default=2000000,
                       help='Total training steps')
    parser.add_argument('--rollout_length', type=int, default=2048,
                       help='Rollout length')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                       help='PPO update epochs')
    parser.add_argument('--mini_batch_size', type=int, default=256,
                       help='Mini-batch size')
    
    # Model parameters
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze VLA backbone')
    parser.add_argument('--sft_checkpoint', type=str, default=None,
                       help='SFT warm-up checkpoint path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_domain_randomization', action='store_true',
                       help='Disable domain randomization')
    parser.add_argument('--stage_window', type=int, default=8,
                       help='Stage gating window size')
    parser.add_argument('--stage_threshold', type=float, default=0.15,
                       help='Stage gating threshold')
    parser.add_argument('--stage_metric', type=str,
                       choices=['delta', 'score'], default='delta',
                       help='Stage gating metric')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=25000,
                       help='Save interval')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f'outputs/{args.task}/ppo'
    
    main(args)

