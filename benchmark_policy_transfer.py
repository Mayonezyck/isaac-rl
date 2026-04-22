"""
Cross-Simulator Policy Transfer Benchmark
==========================================

Evaluates how well MetaDrive's pretrained expert policy transfers zero-shot to SceneFactory.

This benchmark directly tests the hypothesis: "Do simulators with similar physics and driving 
dynamics enable meaningful policy transfer?"

Metrics:
- Success Rate: % of episodes where goal reached
- Collision Rate: % of episodes with collision
- Episode Length: avg steps to goal
- Reward: cumulative reward per episode

Paper Integration:
- Row in tab:policy_transfer or new table
- Comparison: MetaDrive expert on SceneFactory vs SceneFactory expert on SceneFactory (oracle)
- Shows relative platform compatibility/alignment

Usage:
    conda run -n isaac-rl python benchmark_policy_transfer.py \\
        --num-episodes 50 \\
        --env-config scenefactory_config.yaml \\
        --output policy_transfer_results.json
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# This assumes SceneFactory is installed in isaac-rl repo
try:
    from scenefactory.envs import SceneFactoryEnv
    HAS_SCENEFACTORY = True
except ImportError:
    HAS_SCENEFACTORY = False

from metadrive.examples.scenefactory_adapter import SceneFactoryToMetaDriveAdapter


@dataclass
class EpisodeResult:
    """Per-episode transfer benchmark result."""
    episode_id: int
    steps: int
    cumulative_reward: float
    terminated_reason: str  # "success", "collision", "timeout", "error"
    collision_count: int
    max_speed: float
    min_speed: float
    avg_speed: float


def run_transfer_benchmark(
    env,
    num_episodes: int = 50,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run policy transfer benchmark: MetaDrive expert on SceneFactory environment.
    
    Args:
        env: SceneFactory environment instance
        num_episodes: number of episodes to evaluate
        max_steps_per_episode: max steps per episode (timeout)
        deterministic: use deterministic expert policy (mean) vs stochastic (sample)
        verbose: print progress
    
    Returns:
        Dict with aggregate and per-episode results
    """
    adapter = SceneFactoryToMetaDriveAdapter(deterministic=deterministic)
    
    episodes: List[EpisodeResult] = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Cross-Simulator Policy Transfer Benchmark")
        print(f"Policy: MetaDrive Expert (pretrained PPO)")
        print(f"Target: SceneFactory Environment")
        print(f"Episodes: {num_episodes} | Steps/episode: {max_steps_per_episode}")
        print(f"Deterministic: {deterministic}")
        print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for ep_id in range(num_episodes):
        obs, info = env.reset()
        
        episode_reward = 0.0
        collision_count = 0
        step = 0
        terminated_reason = "timeout"
        speeds = []
        
        try:
            for step in range(max_steps_per_episode):
                # --- POLICY TRANSFER ---
                # 1. SceneFactory obs → MetaDrive obs
                metadrive_obs = adapter.scenefactory_to_metadrive(obs)
                
                # 2. Run MetaDrive expert
                metadrive_action = adapter.get_metadrive_expert_action(metadrive_obs)
                
                # 3. Map action back to SceneFactory
                scenefactory_action = adapter.metadrive_to_scenefactory_action(metadrive_action)
                
                # --- STEP ENVIRONMENT ---
                obs, reward, terminated, truncated, info = env.step(scenefactory_action)
                episode_reward += reward
                
                # Track stats
                ego_speed = info.get("ego_speed", 0.0)
                speeds.append(ego_speed)
                
                if info.get("collision", False):
                    collision_count += 1
                
                # Check termination
                if terminated:
                    if info.get("is_success", False):
                        terminated_reason = "success"
                    elif info.get("is_collision", False):
                        terminated_reason = "collision"
                    else:
                        terminated_reason = "other"
                    break
                
                if truncated:
                    terminated_reason = "timeout"
                    break
        
        except Exception as e:
            if verbose:
                print(f"  Episode {ep_id}: ERROR - {str(e)[:50]}")
            terminated_reason = "error"
            step = max_steps_per_episode
        
        # Record episode
        result = EpisodeResult(
            episode_id=ep_id,
            steps=step + 1,
            cumulative_reward=episode_reward,
            terminated_reason=terminated_reason,
            collision_count=collision_count,
            max_speed=max(speeds) if speeds else 0.0,
            min_speed=min(speeds) if speeds else 0.0,
            avg_speed=np.mean(speeds) if speeds else 0.0
        )
        episodes.append(result)
        
        if verbose and (ep_id + 1) % 10 == 0:
            print(f"  Episode {ep_id+1}/{num_episodes} | "
                  f"Reward: {episode_reward:7.2f} | Steps: {step+1:4d} | "
                  f"Reason: {terminated_reason:10s}")
    
    elapsed = time.time() - start_time
    
    # --- AGGREGATE STATISTICS ---
    successes = sum(1 for e in episodes if e.terminated_reason == "success")
    collisions = sum(1 for e in episodes if e.terminated_reason == "collision")
    timeouts = sum(1 for e in episodes if e.terminated_reason == "timeout")
    errors = sum(1 for e in episodes if e.terminated_reason == "error")
    
    rewards = [e.cumulative_reward for e in episodes]
    steps = [e.steps for e in episodes]
    avg_speeds = [e.avg_speed for e in episodes]
    
    results = {
        "benchmark": "policy_transfer",
        "policy": "MetaDrive Expert (PPO, 2-layer)",
        "environment": "SceneFactory",
        "configuration": {
            "num_episodes": num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "deterministic": deterministic
        },
        "aggregate": {
            "success_rate": successes / num_episodes,
            "collision_rate": collisions / num_episodes,
            "timeout_rate": timeouts / num_episodes,
            "error_rate": errors / num_episodes,
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_episode_length": np.mean(steps),
            "std_episode_length": np.std(steps),
            "avg_speed": np.mean(avg_speeds),
            "std_speed": np.std(avg_speeds)
        },
        "per_episode": [asdict(e) for e in episodes],
        "elapsed_time": elapsed,
        "episodes_per_second": num_episodes / elapsed
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results Summary")
        print(f"{'='*70}")
        print(f"Success Rate:        {results['aggregate']['success_rate']:.1%}")
        print(f"Collision Rate:      {results['aggregate']['collision_rate']:.1%}")
        print(f"Timeout Rate:        {results['aggregate']['timeout_rate']:.1%}")
        print(f"Avg Reward:          {results['aggregate']['avg_reward']:.2f} ± {results['aggregate']['std_reward']:.2f}")
        print(f"Avg Episode Length:  {results['aggregate']['avg_episode_length']:.1f} ± {results['aggregate']['std_episode_length']:.1f} steps")
        print(f"Avg Speed:           {results['aggregate']['avg_speed']:.2f} ± {results['aggregate']['std_speed']:.2f} m/s")
        print(f"Total Time:          {elapsed:.1f}s ({results['episodes_per_second']:.1f} eps/s)")
        print(f"{'='*70}\n")
    
    return results


def print_latex_table_row(results: Dict) -> str:
    """
    Generate LaTeX table row for paper integration.
    
    Example output:
        \textbf{MetaDrive Expert} & 50\% & 10\% & \textit{Transfer} & ... \\
    """
    agg = results["aggregate"]
    row = (
        f"MetaDrive Expert $\\to$ SceneFactory"
        f" & {agg['success_rate']:.1%}"
        f" & {agg['collision_rate']:.1%}"
        f" & {agg['avg_reward']:.2f}"
        f" & {agg['avg_episode_length']:.0f}"
        f" & \\textit{{Cross-sim}} \\\\"
    )
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Cross-simulator policy transfer benchmark"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate (default: 50)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (mean)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (sample from distribution)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="policy_transfer_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table row"
    )
    
    args = parser.parse_args()
    
    # Check if SceneFactory is available
    if not HAS_SCENEFACTORY:
        print("ERROR: SceneFactory not found. Install from isaac-rl repository:")
        print("  cd /path/to/isaac-rl")
        print("  pip install -e .")
        return
    
    # Create environment
    print("Initializing SceneFactory environment...")
    try:
        env = SceneFactoryEnv()
    except Exception as e:
        print(f"ERROR initializing SceneFactory: {e}")
        return
    
    # Run benchmark
    deterministic = not args.stochastic
    results = run_transfer_benchmark(
        env,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        deterministic=deterministic,
        verbose=True
    )
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_path}")
    
    # Print LaTeX table row if requested
    if args.latex:
        print("\nLaTeX table row:")
        print(print_latex_table_row(results))
    
    env.close()


if __name__ == "__main__":
    main()
