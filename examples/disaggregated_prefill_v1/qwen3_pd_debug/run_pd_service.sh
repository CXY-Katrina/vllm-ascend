#!/usr/bin/env bash

set -euo pipefail

readonly DEFAULT_MODEL="Qwen/Qwen3-235B-A22B"
readonly DEFAULT_SERVER_PORT=8080
readonly DEFAULT_DP_RPC_PORT=13389
readonly DEFAULT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
readonly REQUIRED_NPU_COUNT=16
readonly PREFILL_DP_SIZE=2
readonly PREFILL_TP_SIZE=8
readonly PREFILL_KV_PORT=30000
readonly DECODE_DP_SIZE=4
readonly DECODE_TP_SIZE=4
readonly DECODE_KV_PORT=30100

usage() {
  cat <<'EOF'
Usage:
  run_pd_service.sh <prefill|decode> --local-ip IP --nic-name NAME [options]

Required:
  --local-ip IP             Host IP reachable from the other PD node.
  --nic-name NAME           Network interface that owns --local-ip.

Options:
  --model MODEL             ModelScope ID or local model path.
  --server-port PORT        OpenAI API port (default: 8080).
  --dp-rpc-port PORT        Internal data-parallel RPC port (default: 13389).
  --visible-devices LIST    Sixteen comma-separated NPU IDs (default: 0..15).
  --kv-port PORT            Mooncake metadata port (default: 30000 for P,
                            30100 for D).
  --dry-run                 Print the environment and command without starting.
  -h, --help                Show this help.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || fail "$option requires a value"
}

validate_port() {
  local name="$1"
  local port="$2"
  [[ "$port" =~ ^[0-9]+$ ]] || fail "$name must be an integer: $port"
  ((port >= 1 && port <= 65535)) || fail "$name must be between 1 and 65535: $port"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

[[ $# -ge 1 ]] || {
  usage >&2
  exit 2
}

role="$1"
shift

case "$role" in
  prefill | decode) ;;
  *) fail "role must be 'prefill' or 'decode': $role" ;;
esac

local_ip=""
nic_name=""
model="$DEFAULT_MODEL"
server_port="$DEFAULT_SERVER_PORT"
dp_rpc_port="$DEFAULT_DP_RPC_PORT"
visible_devices="$DEFAULT_VISIBLE_DEVICES"
kv_port=""
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-ip)
      require_value "$1" "${2:-}"
      local_ip="$2"
      shift 2
      ;;
    --nic-name)
      require_value "$1" "${2:-}"
      nic_name="$2"
      shift 2
      ;;
    --model)
      require_value "$1" "${2:-}"
      model="$2"
      shift 2
      ;;
    --server-port)
      require_value "$1" "${2:-}"
      server_port="$2"
      shift 2
      ;;
    --dp-rpc-port)
      require_value "$1" "${2:-}"
      dp_rpc_port="$2"
      shift 2
      ;;
    --visible-devices)
      require_value "$1" "${2:-}"
      visible_devices="$2"
      shift 2
      ;;
    --kv-port)
      require_value "$1" "${2:-}"
      kv_port="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "$local_ip" ]] || fail "--local-ip is required"
[[ -n "$nic_name" ]] || fail "--nic-name is required"
validate_port "--server-port" "$server_port"
validate_port "--dp-rpc-port" "$dp_rpc_port"

IFS=',' read -r -a npu_devices <<<"$visible_devices"
if ((${#npu_devices[@]} != REQUIRED_NPU_COUNT)); then
  fail "--visible-devices must contain exactly $REQUIRED_NPU_COUNT NPU IDs"
fi

if [[ "$role" == "prefill" ]]; then
  dp_size="$PREFILL_DP_SIZE"
  tp_size="$PREFILL_TP_SIZE"
  kv_role="kv_producer"
  kv_port="${kv_port:-$PREFILL_KV_PORT}"
else
  dp_size="$DECODE_DP_SIZE"
  tp_size="$DECODE_TP_SIZE"
  kv_role="kv_consumer"
  kv_port="${kv_port:-$DECODE_KV_PORT}"
fi
validate_port "--kv-port" "$kv_port"

if [[ "$dry_run" == false ]]; then
  command -v vllm >/dev/null 2>&1 || fail "vllm is not available in PATH"
  command -v ip >/dev/null 2>&1 || fail "ip is not available in PATH"
  ip link show dev "$nic_name" >/dev/null 2>&1 || fail "network interface does not exist: $nic_name"
  if ! ip -o addr show dev "$nic_name" | grep -F -- "$local_ip" >/dev/null; then
    fail "$local_ip is not assigned to network interface $nic_name"
  fi
fi

export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export SERVER_PORT="$server_port"
export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"
export HCCL_IF_IP="$local_ip"
export HCCL_SOCKET_IFNAME="$nic_name"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
export PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-$visible_devices}"

kv_transfer_config=$(printf '%s' \
  '{"kv_connector":"MooncakeConnectorV1",' \
  "\"kv_role\":\"$kv_role\"," \
  "\"kv_port\":\"$kv_port\"," \
  '"kv_connector_extra_config":{' \
  "\"prefill\":{\"dp_size\":$PREFILL_DP_SIZE,\"tp_size\":$PREFILL_TP_SIZE}," \
  "\"decode\":{\"dp_size\":$DECODE_DP_SIZE,\"tp_size\":$DECODE_TP_SIZE}" \
  '}}')

command=(
  vllm serve "$model"
  --host 0.0.0.0
  --port "$server_port"
  --data-parallel-size "$dp_size"
  --data-parallel-size-local "$dp_size"
  --data-parallel-start-rank 0
  --data-parallel-address "$local_ip"
  --data-parallel-rpc-port "$dp_rpc_port"
  --tensor-parallel-size "$tp_size"
  --seed 1024
  --max-num-seqs 32
  --max-model-len 8192
  --max-num-batched-tokens 8192
  --enable-expert-parallel
  --trust-remote-code
  --gpu-memory-utilization 0.9
  --no-enable-prefix-caching
  --additional-config '{"enable_fused_mc2":1}'
)

if [[ "$role" == "decode" ]]; then
  command+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
fi
command+=(--kv-transfer-config "$kv_transfer_config")

if [[ "$dry_run" == true ]]; then
  echo "Role: $role (DP=$dp_size, TP=$tp_size, Mooncake=$kv_role:$kv_port)"
  echo "Environment:"
  for name in \
    VLLM_USE_MODELSCOPE PYTORCH_NPU_ALLOC_CONF HCCL_BUFFSIZE \
    HCCL_OP_EXPANSION_MODE OMP_PROC_BIND OMP_NUM_THREADS \
    TASK_QUEUE_ENABLE VLLM_WORKER_MULTIPROC_METHOD SERVER_PORT \
    LOCAL_IP NIC_NAME HCCL_IF_IP \
    HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME \
    ASCEND_RT_VISIBLE_DEVICES PHYSICAL_DEVICES; do
    printf '  %s=%q\n' "$name" "${!name}"
  done
  echo "Command:"
  print_command "${command[@]}"
  exit 0
fi

echo "Starting $role service on $local_ip:$server_port (DP=$dp_size, TP=$tp_size)."
echo "Press Ctrl+C to stop the service."
exec "${command[@]}"
