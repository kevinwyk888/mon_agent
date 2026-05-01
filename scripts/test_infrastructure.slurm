#!/usr/bin/env bash
#SBATCH --job-name=mon-test
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test-infra-%j.out

# ============================================================
# Quick infrastructure test:
#   1. GPU node access (spgpu + A40)
#   2. Singularity (fakeroot + sandbox build)
#   3. vLLM (load DeepSeek-R1-8B from existing hf_cache)
#   4. Mini-SWE-agent MonitoringAgent import
# ============================================================

set -euo pipefail

PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "========================================"
echo " Infrastructure Test — $(date)"
echo " Node: $(hostname -s)"
echo "========================================"

# --- Test 1: GPU access ---
echo ""
echo "=== Test 1: GPU access ==="
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  pass "GPU accessible"
else
  fail "nvidia-smi failed"
fi

# --- Test 2: Modules ---
echo ""
echo "=== Test 2: Load modules ==="
module load python/3.11.5 && pass "python/3.11.5" || fail "python/3.11.5"
module load cuda/12.1 && pass "cuda/12.1" || fail "cuda/12.1"
module load singularity && pass "singularity" || fail "singularity"
singularity --version && pass "singularity version" || fail "singularity version"

# --- Test 3: Python env ---
echo ""
echo "=== Test 3: Python environment ==="
ENV_DIR="$HOME/.venvs/agent_env"
source "$ENV_DIR/bin/activate"
python --version && pass "python activated" || fail "python activated"

PYTHONPATH="/home/rongbo/Agent/mon-agent/src:${PYTHONPATH:-}"
export PYTHONPATH

python -c "import vllm; print(f'vLLM {vllm.__version__}')" && pass "vllm import" || fail "vllm import"
python -c "import litellm; print(f'litellm {litellm.__version__}')" && pass "litellm import" || fail "litellm import"
python -c "import minisweagent; print(f'mini-swe-agent {minisweagent.__version__}')" && pass "mini-swe-agent import" || fail "mini-swe-agent import"
python -c "from mon_agent.agent import MonitoringAgent; print('MonitoringAgent OK')" && pass "MonitoringAgent import" || fail "MonitoringAgent import"

# --- Test 4: Singularity fakeroot ---
echo ""
echo "=== Test 4: Singularity fakeroot ==="
SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
TEST_SANDBOX="/tmp/${USER}/test_sandbox_${SLURM_JOB_ID}"
mkdir -p "$TEST_SANDBOX"

echo "Building sandbox from alpine..."
if singularity build --sandbox --fakeroot "$TEST_SANDBOX/alpine" docker://alpine:latest 2>&1 | tail -3; then
  pass "singularity build --sandbox --fakeroot"
else
  fail "singularity build --sandbox --fakeroot"
  echo "  Trying without fakeroot..."
  if singularity build --sandbox "$TEST_SANDBOX/alpine_nofr" docker://alpine:latest 2>&1 | tail -3; then
    pass "singularity build --sandbox (no fakeroot)"
  else
    fail "singularity build --sandbox (no fakeroot)"
  fi
fi

if [ -d "$TEST_SANDBOX/alpine" ]; then
  WHOAMI=$(singularity exec --fakeroot --writable "$TEST_SANDBOX/alpine" whoami 2>&1) || true
  echo "  whoami inside container: $WHOAMI"
  if [ "$WHOAMI" = "root" ]; then
    pass "fakeroot gives root inside container"
  else
    fail "fakeroot whoami returned: $WHOAMI"
  fi
fi
rm -rf "$TEST_SANDBOX"

# --- Test 5: Network access (for pulling SWE-bench images) ---
echo ""
echo "=== Test 5: Network access ==="
if curl -s --max-time 10 https://hub.docker.com/v2/ >/dev/null 2>&1; then
  pass "Docker Hub reachable"
else
  fail "Docker Hub NOT reachable from compute node"
fi

# --- Test 6: vLLM server (quick start + health check) ---
echo ""
echo "=== Test 6: vLLM server ==="
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
HF_CACHE="${SCRATCH_BASE}/hf_cache"
PORT=8199

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

echo "Starting vLLM with ${MODEL}..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir "$HF_CACHE/hub" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 &

VLLM_PID=$!

echo "Waiting for vLLM (max 180s)..."
READY=false
for i in $(seq 1 180); do
  if curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
    echo "  Server ready after ${i}s"
    READY=true
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "  vLLM process died"
    break
  fi
  sleep 1
done

if $READY; then
  pass "vLLM server started"

  # Quick inference test
  echo "  Running inference test..."
  RESPONSE=$(curl -s http://127.0.0.1:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"$MODEL"'",
      "messages": [{"role": "user", "content": "Say hello in one word."}],
      "max_tokens": 16,
      "temperature": 0.0
    }' 2>&1)

  if echo "$RESPONSE" | python -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null; then
    pass "vLLM inference works"
  else
    fail "vLLM inference failed: $RESPONSE"
  fi
else
  fail "vLLM server did not start"
fi

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# --- Summary ---
echo ""
echo "========================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [ $FAIL -gt 0 ]; then
  exit 1
fi
