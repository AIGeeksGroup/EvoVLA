#!/usr/bin/env python3
"""
Generate Stage Triplets using LLM APIs

This script generates triplet text descriptions (positive, negative, hard_negative)
for each stage of a manipulation task using LLM APIs (Gemini, DeepSeek, or OpenAI).
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional


def get_llm_client(provider: str = "openai"):
    """Get LLM client based on provider."""
    if provider == "deepseek":
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), "deepseek-chat"
    
    elif provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key), "gpt-4"
    
    elif provider == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro'), "gemini-pro"
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_triplet(
    client,
    model_name: str,
    task_name: str,
    stage_idx: int,
    positive_desc: str,
    provider: str = "openai"
) -> Dict[str, str]:
    """
    Generate triplet for a single stage.
    
    Args:
        client: LLM client
        model_name: Model name
        task_name: Task name
        stage_idx: Stage index
        positive_desc: Positive description (from code comments)
        provider: LLM provider
    
    Returns:
        dict: {"positive": str, "negative": str, "hard_negative": str}
    """
    prompt = f"""You are a robotic vision-language expert. Generate triplet text descriptions for a manipulation task stage.

Task: {task_name}
Current Stage (Stage {stage_idx}): {positive_desc}

Generate three descriptions using **spatial/contact/geometric predicates**, avoiding color/texture visual features:

1. **positive**: Use the current stage description, or expand it to be more explicit
2. **negative (mutually exclusive)**: Describe a completely different state (e.g., gripper far from objects, completely different action)
3. **hard_negative (counterfactual near-miss)**: Describe a near-miss state that is very close to positive but critically different
   - Use spatial predicates: e.g., "near non-target object", "below object instead of above"
   - Use contact predicates: e.g., "touching but not grasping", "open instead of closed"
   - Avoid colors: don't say "grasping blue cup instead of red", use "grasping non-target object"

Example format:
{{
  "positive": "Robot gripper approaching target bar, fingers open ready to grasp",
  "negative": "Robot arm far from all objects, gripper closed at initial position",
  "hard_negative": "Robot gripper approaching non-target object, fingers open"
}}

Output JSON format only, with positive, negative, hard_negative fields.
"""
    
    try:
        if provider == "gemini":
            response = client.generate_content(prompt)
            result_text = response.text
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a robotic manipulation expert, skilled at describing robot states using spatial/contact predicates, avoiding reliance on visual appearance."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        
        return result
    
    except Exception as e:
        print(f"  ✗ Stage {stage_idx} generation failed: {e}")
        return {
            "positive": positive_desc,
            "negative": "Robot arm at initial position, not started",
            "hard_negative": f"Robot arm performing similar but incorrect action"
        }


def generate_all_triplets(
    stages_file: str,
    output_dir: str,
    provider: str = "openai"
):
    """Generate triplets for all tasks and stages."""
    
    # Load extracted stages
    with open(stages_file, 'r', encoding='utf-8') as f:
        tasks_data = json.load(f)
    
    client, model_name = get_llm_client(provider)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task_key, task_info in tasks_data.items():
        print(f"\n{'='*60}")
        print(f"Processing task: {task_info['task_name']} ({task_info['num_stages']} stages)")
        print(f"{'='*60}")
        
        triplets = {}
        stages = task_info['stages']
        
        for stage_idx_str, stage_desc in stages.items():
            stage_idx = int(stage_idx_str)
            print(f"  Generating Stage {stage_idx:2d}: {stage_desc[:40]}...", end=" ")
            
            triplet = generate_triplet(
                client,
                model_name,
                task_info['task_description'],
                stage_idx,
                stage_desc,
                provider
            )
            
            triplets[f"stage_{stage_idx}"] = triplet
            print("✓")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save triplets for this task
        task_output_file = output_dir / f"{task_key}_triplets.json"
        with open(task_output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "task_name": task_info['task_name'],
                "task_description": task_info['task_description'],
                "num_stages": len(triplets),
                "triplets": triplets
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved: {task_output_file}")
    
    print(f"\n{'='*60}")
    print("✅ All triplets generated!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Stage Triplets')
    parser.add_argument('--stages_file', type=str, required=True,
                       help='Path to extracted stages JSON file')
    parser.add_argument('--output_dir', type=str, default='stage_dictionaries',
                       help='Output directory for triplets')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'deepseek', 'gemini'],
                       help='LLM provider')
    
    args = parser.parse_args()
    
    generate_all_triplets(args.stages_file, args.output_dir, args.provider)

