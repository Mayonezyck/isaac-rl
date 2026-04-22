#!/bin/bash
# Cross-Simulator Policy Transfer: Quick Start Guide
# ===================================================
#
# This script sets up and runs the policy transfer benchmark.
# 
# Usage:
#   bash run_policy_transfer.sh
#   bash run_policy_transfer.sh --num-episodes 100 --output results.json

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Cross-Simulator Policy Transfer Benchmark - Quick Start           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Default parameters
NUM_EPISODES=50
OUTPUT_FILE="policy_transfer_results.json"
SHOW_LATEX=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --no-latex)
            SHOW_LATEX=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num-episodes N] [--output FILE] [--no-latex]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Episodes:      $NUM_EPISODES"
echo "  Output file:   $OUTPUT_FILE"
echo "  LaTeX output:  $SHOW_LATEX"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check MetaDrive
echo -n "  MetaDrive: "
if python3 -c "import metadrive; print(f'v{metadrive.__version__}')" 2>/dev/null; then
    :
else
    echo "❌ NOT FOUND"
    echo ""
    echo "ERROR: MetaDrive not installed. Install with:"
    echo "  cd /home/yz8733/Github/metadrive"
    echo "  pip install -e ."
    exit 1
fi

# Check MetaDrive adapter
echo -n "  SceneFactory Adapter: "
if python3 -c "from metadrive.examples.scenefactory_adapter import SceneFactoryToMetaDriveAdapter; print('✓')" 2>/dev/null; then
    :
else
    echo "❌ NOT FOUND"
    echo ""
    echo "ERROR: Adapter not found. Ensure scenefactory_adapter.py is in:"
    echo "  /home/yz8733/Github/metadrive/metadrive/examples/"
    exit 1
fi

# Check expert weights
echo -n "  Expert weights: "
if [[ -f "/home/yz8733/Github/metadrive/metadrive/examples/ppo_expert/expert_weights.npz" ]]; then
    echo "✓"
else
    echo "❌ NOT FOUND"
    echo ""
    echo "ERROR: Expert weights not found at:"
    echo "  /home/yz8733/Github/metadrive/metadrive/examples/ppo_expert/expert_weights.npz"
    exit 1
fi

# Check SceneFactory (optional but strongly recommended)
echo -n "  SceneFactory: "
if python3 -c "from scenefactory.envs import SceneFactoryEnv; print('✓')" 2>/dev/null; then
    :
else
    echo "⚠️ NOT FOUND (optional)"
    echo "    Note: SceneFactory required for actual benchmark evaluation"
    echo "    To install: cd /home/yz8733/Github/isaac-rl && pip install -e ."
    echo ""
fi

echo ""
echo "✓ Prerequisites check complete"
echo ""

# Test adapter
echo "Testing adapter..."
python3 << 'EOF'
import sys
try:
    from metadrive.examples.scenefactory_adapter import SceneFactoryToMetaDriveAdapter
    import numpy as np
    
    # Initialize adapter
    adapter = SceneFactoryToMetaDriveAdapter(deterministic=True)
    print("  ✓ Adapter initialized")
    
    # Test dummy conversion
    dummy_obs = np.random.randn(1929).astype(np.float32)
    dummy_ego = np.array([0, 0, 10, 0, 0, 0, 0, 0, 2.7, 1.0, 0], dtype=np.float32)
    
    md_obs = adapter.scenefactory_to_metadrive(dummy_obs, dummy_ego)
    print(f"  ✓ Observation conversion: {dummy_obs.shape} → {md_obs.shape}")
    
    # Test expert
    md_action = adapter.get_metadrive_expert_action(md_obs)
    print(f"  ✓ Expert inference: {md_action.shape} action generated")
    
    # Test action mapping
    sf_action = adapter.metadrive_to_scenefactory_action(md_action)
    print(f"  ✓ Action mapping: {md_action.shape} → {sf_action.shape}")
    
    print("\n✓ Adapter tests passed")
    
except Exception as e:
    print(f"\n❌ Adapter test failed: {e}")
    sys.exit(1)
EOF

if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ Adapter tests failed"
    exit 1
fi

echo ""

# Try to run full benchmark
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running policy transfer benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /home/yz8733/Github/isaac-rl

# Build command
CMD="python benchmark_policy_transfer.py --num-episodes $NUM_EPISODES --output $OUTPUT_FILE --deterministic"

if [[ "$SHOW_LATEX" == true ]]; then
    CMD="$CMD --latex"
fi

# Run benchmark
if python3 $CMD; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✓ Benchmark completed successfully!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_FILE | python3 -m json.tool"
    echo ""
    echo "For LaTeX table row, re-run with --latex flag:"
    echo "  bash run_policy_transfer.sh --num-episodes $NUM_EPISODES --latex"
    echo ""
else
    echo ""
    echo "❌ Benchmark execution failed"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check that SceneFactory is properly installed"
    echo "  2. Verify all imports are available"
    echo "  3. Check GPU memory and availability"
    echo ""
    exit 1
fi

exit 0
