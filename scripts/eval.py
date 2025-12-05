#!/usr/bin/env python3
"""
Evaluation Script for EvoVLA

Features:
1. Load trained policy
2. Run evaluation episodes
3. Compute metrics (Success Rate, Sample Efficiency, Hallucination Rate)
4. Save results
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evovla.models.openvla_policy_oft import OpenVLAPolicyOFT
from evovla.rewards.stage_aligned_reward import StageAlignedReward
from evovla.data.discoverse_env import DiscoverseEnv


def evaluate(policy, env, sar, num_episodes: int = 50):
    """
    Evaluate policy (paper standard, with per-stage statistics).
    
    Args:
        policy: OpenVLAPolicyOFT instance
        env: DiscoverseEnv instance
        sar: StageAlignedReward instance
        num_episodes: Number of evaluation episodes
    
    Returns:
        results: Evaluation results dictionary
    """
    successes = 0
    total_steps = 0
    all_rewards = []
    all_steps_per_episode = []
    
    # Hallucination rate statistics
    high_vlm_scores = 0
    hallucinations = 0
    
    # Per-stage statistics
    stage_attempts = {}
    stage_completions = {}
    
    # Action smoothness
    action_changes = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        sar.reset()
        
        episode_reward = 0
        episode_steps = 0
        done = False
        max_steps = 400
        
        prev_action = None
        episode_action_changes = []
        
        while not done and episode_steps < max_steps:
            # Use forward_train for inference (uses trained action_head_sft)
            obs_dict = {
                'image': obs['image'],
                'task_description': obs['task_description']
            }
            mu, log_std, v_ext, v_int = policy.forward_train(obs_dict)
            action_tensor = policy.decode_from_pre_activation(mu)
            action = action_tensor.detach().cpu().numpy()
            
            # Action smoothness
            if prev_action is not None:
                action_change = np.linalg.norm(action - prev_action)
                episode_action_changes.append(action_change)
            prev_action = action.copy()
            
            # Per-stage statistics
            current_stage = env.current_stage
            stage_attempts[current_stage] = stage_attempts.get(current_stage, 0) + 1
            
            # SAR score for hallucination rate
            sar_image = obs.get('full_image')
            if sar_image is None:
                base_image = obs['image']
                if isinstance(base_image, Image.Image):
                    sar_image = np.array(base_image)
                else:
                    sar_image = base_image
            
            r_stage, u_k = sar.compute(sar_image, current_stage)
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Stage gating
            advanced, old_stage, new_stage = env.update_stage_gating(r_stage, u_k)
            if advanced:
                sar.reset()
            
            # Hallucination detection
            if u_k > 0.7:
                high_vlm_scores += 1
                if r_stage < 0.1:
                    hallucinations += 1
            
            obs = next_obs
        
        # Record episode statistics
        if info.get('success', False):
            successes += 1
            max_stage = info.get('current_stage', 0)
            stage_completions[max_stage] = stage_completions.get(max_stage, 0) + 1
        
        all_rewards.append(episode_reward)
        all_steps_per_episode.append(episode_steps)
        total_steps += episode_steps
        
        if episode_action_changes:
            action_changes.extend(episode_action_changes)
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"reward={episode_reward:.2f}, steps={episode_steps}, "
              f"success={info.get('success', False)}")
    
    # Compute metrics
    success_rate = float(successes / num_episodes)
    avg_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    avg_steps = float(np.mean(all_steps_per_episode))
    std_steps = float(np.std(all_steps_per_episode))
    hallucination_rate = float(hallucinations / max(high_vlm_scores, 1))
    
    # Per-stage success rates
    stage_success_rates = {}
    for stage_id in sorted(set(list(stage_attempts.keys()) + list(stage_completions.keys()))):
        completion_count = stage_completions.get(stage_id, 0)
        attempt_count = stage_attempts.get(stage_id, 0)
        if attempt_count > 0:
            stage_success_rates[stage_id] = float(completion_count / attempt_count)
        else:
            stage_success_rates[stage_id] = 0.0
    
    avg_action_change = float(np.mean(action_changes)) if action_changes else 0.0
    
    results = {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_steps': avg_steps,
        'std_steps': std_steps,
        'hallucination_rate': hallucination_rate,
        'num_episodes': int(num_episodes),
        'successes': int(successes),
        'stage_success_rates': stage_success_rates,
        'stage_attempts': stage_attempts,
        'stage_completions': stage_completions,
        'avg_action_change': avg_action_change,
        'success_rate_ci': float(1.96 * np.sqrt(success_rate * (1 - success_rate) / num_episodes)),
        'reward_ci': float(1.96 * std_reward / np.sqrt(num_episodes)),
    }
    
    return results


def main(args):
    print("=" * 80)
    print(f"Evaluating task: {args.task}")
    print("=" * 80)
    
    # 1. Load policy
    print("\nLoading policy...")
    policy = OpenVLAPolicyOFT(
        model_path=args.model_path,
        device="cuda",
        mode="ppo",
        freeze_backbone=True,
        enable_lora=args.enable_lora or (args.lora_adapter_path is not None),
        lora_adapter_path=args.lora_adapter_path,
    )
    
    policy.set_task_norm_stats(args.task)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        has_action_head = 'action_head_sft_state_dict' in ckpt
        
        if has_action_head:
            print("Initializing action_head_sft...")
            dummy_image = Image.new('RGB', (448, 448), color=(128, 128, 128))
            dummy_obs = {'image': dummy_image, 'task_description': "dummy task"}
            policy.eval()
            _ = policy.forward_train(dummy_obs)
        
        policy.load_for_inference(args.checkpoint)
        print(f"✓ Policy loaded (action_head_sft: {'✅' if has_action_head else '❌'})")
    
    policy.eval()
    
    # 2. Load SAR
    print("\nLoading SAR...")
    triplets_path = args.triplets or f'stage_dictionaries/{args.task}_triplets.json'
    sar = StageAlignedReward(triplets_path, device="cuda")
    print(f"✓ SAR loaded")
    
    # 3. Create environment
    print("\nCreating environment...")
    env = DiscoverseEnv(
        args.task,
        headless=True,
        randomize=args.enable_domain_randomization,
        stage_window_size=args.stage_window,
        stage_completion_threshold=args.stage_threshold,
        stage_metric=args.stage_metric,
        enable_relaxed_success=not args.disable_relaxed_success,
    )
    print(f"✓ Environment created")
    
    # 4. Evaluate
    print("\n" + "=" * 80)
    print(f"Starting evaluation ({args.num_episodes} episodes)...")
    print("=" * 80)
    
    results = evaluate(policy, env, sar, num_episodes=args.num_episodes)
    
    # 5. Print results
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"  Success Rate:        {results['success_rate']:.1%} ± {results['success_rate_ci']:.1%} ({results['successes']}/{results['num_episodes']})")
    print(f"  Average Reward:      {results['avg_reward']:.2f} ± {results['reward_ci']:.2f}")
    print(f"  Average Steps:       {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"  Hallucination Rate:  {results['hallucination_rate']:.1%}")
    print(f"  Action Smoothness:   {results['avg_action_change']:.4f}")
    print(f"\n  Per-Stage Success Rates:")
    for stage_id in sorted(results['stage_success_rates'].keys()):
        rate = results['stage_success_rates'][stage_id]
        attempts = results['stage_attempts'].get(stage_id, 0)
        completions = results['stage_completions'].get(stage_id, 0)
        print(f"    Stage {stage_id}: {rate:.1%} ({completions}/{attempts})")
    
    # 6. Save results
    output_dir = Path(args.output_dir)
    output_file = Path(args.output) if args.output else output_dir / 'eval_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")
    
    env.close()
    print("\n" + "=" * 80)
    print("🎉 Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EvoVLA Evaluation')
    
    parser.add_argument('--task', type=str, required=True,
                       choices=['bridge', 'jujube_cup', 'stack'],
                       help='Task name')
    parser.add_argument('--model_path', type=str,
                       default='/path/to/openvla-7b-oft',
                       help='OpenVLA-OFT model path')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Trained checkpoint path')
    parser.add_argument('--triplets', type=str, default=None,
                       help='Triplets file path')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--enable_domain_randomization', action='store_true',
                       help='Enable domain randomization')
    parser.add_argument('--enable_lora', action='store_true',
                       help='Enable LoRA adapter')
    parser.add_argument('--lora_adapter_path', type=str, default=None,
                       help='LoRA adapter path')
    parser.add_argument('--stage_window', type=int, default=6,
                       help='Stage gating window size')
    parser.add_argument('--stage_threshold', type=float, default=0.015,
                       help='Stage gating threshold')
    parser.add_argument('--stage_metric', type=str,
                       choices=['delta', 'score'], default='delta',
                       help='Stage gating metric')
    parser.add_argument('--disable_relaxed_success', action='store_true',
                       help='Disable relaxed success criteria')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f'outputs/{args.task}/eval'
    
    main(args)

