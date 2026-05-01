#!/usr/bin/env bash

detect_container_runtime() {
  if command -v singularity >/dev/null 2>&1; then
    echo singularity
    return 0
  fi
  if command -v apptainer >/dev/null 2>&1; then
    echo apptainer
    return 0
  fi
  echo "ERROR: singularity/apptainer not found" >&2
  return 1
}

set_container_runtime_paths() {
  : "${SCRATCH_BASE:?SCRATCH_BASE must be set before calling set_container_runtime_paths}"
  : "${NODE_TMP:?NODE_TMP must be set before calling set_container_runtime_paths}"

  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${SCRATCH_BASE}/singularity_cache}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SINGULARITY_CACHEDIR}"
  export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${NODE_TMP}/singularity_tmp}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SINGULARITY_TMPDIR}"

  mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
}

export_container_env_var() {
  local name="$1"
  local value="$2"
  export "SINGULARITYENV_${name}=${value}"
  export "APPTAINERENV_${name}=${value}"
}

set_default_vllm_container_paths() {
  : "${SCRATCH_BASE:?SCRATCH_BASE must be set before calling set_default_vllm_container_paths}"
  VLLM_CONTAINER_REF="${VLLM_CONTAINER_REF:-docker://vllm/vllm-openai:gemma4}"
  local tag="${VLLM_CONTAINER_REF##*:}"
  VLLM_CONTAINER_DIR="${VLLM_CONTAINER_DIR:-${SCRATCH_BASE}/singularity_images}"
  VLLM_CONTAINER_SIF="${VLLM_CONTAINER_SIF:-${VLLM_CONTAINER_DIR}/vllm-openai-${tag}.sif}"
  export VLLM_CONTAINER_REF VLLM_CONTAINER_DIR VLLM_CONTAINER_SIF
}

ensure_vllm_container_image() {
  local runtime="${1:-$(detect_container_runtime)}"
  set_default_vllm_container_paths
  mkdir -p "$VLLM_CONTAINER_DIR"
  if [ -f "$VLLM_CONTAINER_SIF" ]; then
    echo "Using cached vLLM image: $VLLM_CONTAINER_SIF"
    return 0
  fi

  local tmp_sif="${VLLM_CONTAINER_SIF}.tmp.$$"
  rm -f "$tmp_sif"
  echo "Pulling $VLLM_CONTAINER_REF"
  "$runtime" pull "$tmp_sif" "$VLLM_CONTAINER_REF"
  mv "$tmp_sif" "$VLLM_CONTAINER_SIF"
}

set_vllm_container_env() {
  : "${HF_HOME:?HF_HOME must be set}"
  : "${HF_HUB_CACHE:?HF_HUB_CACHE must be set}"
  : "${TMPDIR:?TMPDIR must be set}"
  : "${XDG_CACHE_HOME:?XDG_CACHE_HOME must be set}"
  : "${CUDA_CACHE_PATH:?CUDA_CACHE_PATH must be set}"

  export_container_env_var HF_HOME "$HF_HOME"
  export_container_env_var HF_HUB_CACHE "$HF_HUB_CACHE"
  export_container_env_var TMPDIR "$TMPDIR"
  export_container_env_var XDG_CACHE_HOME "$XDG_CACHE_HOME"
  export_container_env_var CUDA_CACHE_PATH "$CUDA_CACHE_PATH"
  export_container_env_var VLLM_NO_USAGE_STATS "${VLLM_NO_USAGE_STATS:-1}"
  export_container_env_var VLLM_DO_NOT_TRACK "${VLLM_DO_NOT_TRACK:-1}"

  if [ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
    export_container_env_var TORCHINDUCTOR_CACHE_DIR "$TORCHINDUCTOR_CACHE_DIR"
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    export_container_env_var HF_TOKEN "$HF_TOKEN"
  fi

  # Unset host compiler paths so the container uses its own gcc/nvcc
  unset SINGULARITYENV_CC SINGULARITYENV_CXX
  unset APPTAINERENV_CC APPTAINERENV_CXX
  export_container_env_var CC /usr/bin/gcc
  export_container_env_var CXX /usr/bin/g++

  # Pass SSL certs so the container can reach HuggingFace Hub
  if [ -n "${SSL_CERT_FILE:-}" ] && [ -f "$SSL_CERT_FILE" ]; then
    export_container_env_var SSL_CERT_FILE "$SSL_CERT_FILE"
    export VLLM_CONTAINER_EXTRA_BINDS="${SSL_CERT_FILE}:${SSL_CERT_FILE}:ro"
  else
    export VLLM_CONTAINER_EXTRA_BINDS=""
  fi
}
