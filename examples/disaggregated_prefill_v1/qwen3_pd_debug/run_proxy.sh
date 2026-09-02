#!/usr/bin/env bash

set -euo pipefail

readonly DEFAULT_BACKEND_PORT=8080
readonly DEFAULT_PROXY_PORT=8000

usage() {
  cat <<'EOF'
Usage:
  run_proxy.sh --proxy-ip IP --prefill-ip IP --decode-ip IP [options]

Required:
  --proxy-ip IP             IP on which the proxy listens, normally the P node.
  --prefill-ip IP           Reachable IP of the Prefill service.
  --decode-ip IP            Reachable IP of the Decode service.

Options:
  --backend-port PORT       P and D API port (default: 8080).
  --proxy-port PORT         Client-facing proxy port (default: 8000).
  --python PATH             Python executable (default: python3).
  --dry-run                 Print the command without starting.
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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
proxy_script="$repo_root/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"

proxy_ip=""
prefill_ip=""
decode_ip=""
backend_port="$DEFAULT_BACKEND_PORT"
proxy_port="$DEFAULT_PROXY_PORT"
python_bin="python3"
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy-ip)
      require_value "$1" "${2:-}"
      proxy_ip="$2"
      shift 2
      ;;
    --prefill-ip)
      require_value "$1" "${2:-}"
      prefill_ip="$2"
      shift 2
      ;;
    --decode-ip)
      require_value "$1" "${2:-}"
      decode_ip="$2"
      shift 2
      ;;
    --backend-port)
      require_value "$1" "${2:-}"
      backend_port="$2"
      shift 2
      ;;
    --proxy-port)
      require_value "$1" "${2:-}"
      proxy_port="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2:-}"
      python_bin="$2"
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

[[ -n "$proxy_ip" ]] || fail "--proxy-ip is required"
[[ -n "$prefill_ip" ]] || fail "--prefill-ip is required"
[[ -n "$decode_ip" ]] || fail "--decode-ip is required"
validate_port "--backend-port" "$backend_port"
validate_port "--proxy-port" "$proxy_port"
[[ -f "$proxy_script" ]] || fail "proxy script not found: $proxy_script"

if [[ "$dry_run" == false ]]; then
  command -v "$python_bin" >/dev/null 2>&1 || fail "$python_bin is not available in PATH"
fi

command=(
  "$python_bin" "$proxy_script"
  --host "$proxy_ip"
  --port "$proxy_port"
  --prefiller-hosts "$prefill_ip"
  --prefiller-ports "$backend_port"
  --decoder-hosts "$decode_ip"
  --decoder-ports "$backend_port"
)

if [[ "$dry_run" == true ]]; then
  printf 'Command:\n  '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

echo "Starting PD proxy on http://$proxy_ip:$proxy_port."
echo "Prefill: $prefill_ip:$backend_port; Decode: $decode_ip:$backend_port."
echo "Press Ctrl+C to stop the proxy."
exec "${command[@]}"
