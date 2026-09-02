#!/usr/bin/env bash

set -euo pipefail

readonly DEFAULT_BACKEND_PORT=8080
readonly DEFAULT_PROXY_PORT=8000

usage() {
  cat <<'EOF'
Usage:
  check_pd.sh --prefill-ip IP --decode-ip IP --proxy-ip IP [options]

Options:
  --backend-port PORT       P and D API port (default: 8080).
  --proxy-port PORT         Client-facing proxy port (default: 8000).
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

prefill_ip=""
decode_ip=""
proxy_ip=""
backend_port="$DEFAULT_BACKEND_PORT"
proxy_port="$DEFAULT_PROXY_PORT"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --proxy-ip)
      require_value "$1" "${2:-}"
      proxy_ip="$2"
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
    -h | --help)
      usage
      exit 0
      ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "$prefill_ip" ]] || fail "--prefill-ip is required"
[[ -n "$decode_ip" ]] || fail "--decode-ip is required"
[[ -n "$proxy_ip" ]] || fail "--proxy-ip is required"
validate_port "--backend-port" "$backend_port"
validate_port "--proxy-port" "$proxy_port"
command -v curl >/dev/null 2>&1 || fail "curl is not available in PATH"

check_url() {
  local name="$1"
  local url="$2"
  printf 'Checking %-16s %s ... ' "$name" "$url"
  curl --fail --silent --show-error --max-time 10 "$url" >/dev/null
  echo "ok"
}

check_url "Prefill" "http://$prefill_ip:$backend_port/health"
check_url "Decode" "http://$decode_ip:$backend_port/health"
check_url "Proxy scheduler" "http://$proxy_ip:$proxy_port/healthcheck"
check_url "Proxy models" "http://$proxy_ip:$proxy_port/v1/models"

echo "PD service is reachable through http://$proxy_ip:$proxy_port/v1"
