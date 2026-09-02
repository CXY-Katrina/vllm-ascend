# Qwen3-235B PD local debug launcher

These scripts reproduce the service topology from
[`Qwen3-235B-disagg-pd.yaml`](../../../tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-disagg-pd.yaml)
without starting pytest or AISBench. They keep the nightly configuration's model,
parallel layout, environment, graph mode, and Mooncake connector settings.

## Topology and prerequisites

This is a two-host deployment, not a two-NPU single-host example:

| Role | Host | Parallel layout | NPU count | API port | Mooncake port |
| --- | --- | --- | ---: | ---: | ---: |
| Prefill | node 0 | DP 2 x TP 8 | 16 | 8080 | 30000 (base) |
| Decode | node 1 | DP 4 x TP 4 | 16 | 8080 | 30100 (base) |
| Proxy | node 0 | one process | 0 | 8000 | n/a |

Before starting:

- Use the same vLLM and vLLM Ascend revisions on both hosts.
- Run inside an Ascend environment with 16 visible NPUs per host and Mooncake
  dependencies installed.
- Make the model `Qwen/Qwen3-235B-A22B`, or the same local model directory,
  accessible on both hosts.
- Choose host IPs on the same reachable network. Do not use a per-host Docker
  bridge address.
- Ensure the selected NIC is configured for HCCL and that the API, Mooncake,
  and NPU-side communication paths are allowed by the firewall.

Mooncake assigns one handshake port per worker from each configured base port.
For this 16-NPU layout, allow TCP 30000-30015 on Prefill and 30100-30115 on
Decode, in addition to the API/proxy ports required by your client topology.

The examples below use `10.0.0.10` for Prefill, `10.0.0.11` for Decode, and
`eth0` on both hosts. Replace them with addresses and NIC names from your
environment.

## 1. Inspect the commands without starting

On either host, the dry-run mode does not require vLLM, an NPU, or the named
network interface:

```bash
bash examples/disaggregated_prefill_v1/qwen3_pd_debug/run_pd_service.sh \
  prefill --local-ip 10.0.0.10 --nic-name eth0 --dry-run

bash examples/disaggregated_prefill_v1/qwen3_pd_debug/run_pd_service.sh \
  decode --local-ip 10.0.0.11 --nic-name eth0 --dry-run
```

Use `--model /path/to/Qwen3-235B-A22B` on both commands when the model is
already downloaded locally. Use `--visible-devices` if the 16 NPU IDs are not
`0,1,...,15`.

## 2. Start Decode on node 1

```bash
cd /path/to/vllm-ascend
bash examples/disaggregated_prefill_v1/qwen3_pd_debug/run_pd_service.sh \
  decode \
  --local-ip 10.0.0.11 \
  --nic-name eth0 \
  --model /path/to/Qwen3-235B-A22B \
  2>&1 | tee /tmp/qwen3-pd-decode.log
```

Wait until `curl http://10.0.0.11:8080/health` succeeds. Initial model loading
and graph capture can take many minutes.

## 3. Start Prefill on node 0

In a second terminal on node 0:

```bash
cd /path/to/vllm-ascend
bash examples/disaggregated_prefill_v1/qwen3_pd_debug/run_pd_service.sh \
  prefill \
  --local-ip 10.0.0.10 \
  --nic-name eth0 \
  --model /path/to/Qwen3-235B-A22B \
  2>&1 | tee /tmp/qwen3-pd-prefill.log
```

Wait until `curl http://10.0.0.10:8080/health` succeeds.

## 4. Start the request proxy on node 0

In a third terminal on node 0:

```bash
cd /path/to/vllm-ascend
bash examples/disaggregated_prefill_v1/qwen3_pd_debug/run_proxy.sh \
  --proxy-ip 10.0.0.10 \
  --prefill-ip 10.0.0.10 \
  --decode-ip 10.0.0.11 \
  2>&1 | tee /tmp/qwen3-pd-proxy.log
```

Run the bundled connectivity check from node 0 or from the actual client host:

```bash
bash examples/disaggregated_prefill_v1/qwen3_pd_debug/check_pd.sh \
  --proxy-ip 10.0.0.10 \
  --prefill-ip 10.0.0.10 \
  --decode-ip 10.0.0.11
```

## 5. Send a test request

Send client traffic only to the proxy, not directly to either backend:

```bash
curl http://10.0.0.10:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/Qwen3-235B-A22B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

The request's `model` must match the name shown by
`curl http://10.0.0.10:8000/v1/models`. If the services were started with the
default ModelScope ID instead of a local path, use `Qwen/Qwen3-235B-A22B`.

## Stop and troubleshoot

Press `Ctrl+C` in this order: proxy, Prefill, Decode. The scripts run in the
foreground and replace themselves with the target process, so the signal is
delivered directly. After stopping, verify ports 8000 and 8080 are closed and
that no vLLM worker remains before starting another run.

For startup failures, check the role log first, then verify:

- all 16 configured NPUs are visible;
- `DP x TP` is 16 for both roles;
- `--local-ip` belongs to `--nic-name` inside the runtime/container;
- node 0 and node 1 can reach each other's API and Mooncake ports;
- both roles use identical `prefill: DP2/TP8` and `decode: DP4/TP4` connector
  metadata;
- only Decode enables `FULL_DECODE_ONLY` graph mode.
