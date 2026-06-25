# MiniMax M3 Sparse Attention OOB Analysis

Date: 2026-06-24

## Summary

The service was successfully started with `max_model_len=3072`, but the precision test request caused a runtime crash in MiniMax M3 sparse attention.

The failure is not caused by the `npu_clipped_swiglu` MLP fusion itself. The request reached model forward and failed in the sparse attention indexer with an out-of-bounds access:

```text
IndexError: index 306 is out of bounds for dimension 0 with size 306
```

## Service Log Source

Full service log inside container:

```text
/tmp/minimax_m3_bf16_service.log
```

## Runtime Parameters From The Failed Service Log

The failed service run used these effective parameters from the service log:

```text
max_model_len: 3072
tensor_parallel_size: 16
enable_expert_parallel: true
max_num_seqs: 16
gpu_memory_utilization: 0.96
cudagraph_capture_sizes: [1, 2, 4, 8, 16]
port: 11223
```

Important log excerpt:

```text
17:(APIServer pid=2294868) INFO 06-24 21:44:26 [utils.py:240] non-default args: {... 'max_model_len': 3072, ... 'tensor_parallel_size': 16, 'enable_expert_parallel': True, 'gpu_memory_utilization': 0.96, ... 'max_num_seqs': 16, ... 'cudagraph_capture_sizes': [1, 2, 4, 8, 16], ...}
```

Note: when this markdown was generated, the current script files had already changed from the values used in the above log:

```text
/home/cxy/minimax_m3/start_service_bf16.sh:
  --max-model-len 46080
  --max-num-seqs 16
  --gpu-memory-utilization 0.92
  cudagraph_capture_sizes: [1,2,4,8,16]

/home/cxy/minimax_m3/curl_with_throughput.sh:
  MAX_TOKENS="${MAX_TOKENS:-5120}"
```

## Service Startup Result

With `max_model_len=3072`, service startup passed KV cache initialization and graph capture:

```text
1139:(Worker_TP0_EP0 pid=2295927) INFO 06-24 21:47:18 [worker.py:423] [vllm-ascend] - Available KV cache memory: 0.30 GiB
1156:(EngineCore pid=2295872) INFO 06-24 21:47:20 [kv_cache_utils.py:1710] GPU KV cache size: 5,248 tokens
1157:(EngineCore pid=2295872) INFO 06-24 21:47:20 [kv_cache_utils.py:1711] Maximum concurrency for 3,072 tokens per request: 1.71x
1189:(Worker_TP0_EP0 pid=2295927) INFO 06-24 21:47:27 [gpu_model_runner.py:6243] Graph capturing finished in 7 secs, took 0.34 GiB
1355:(APIServer pid=2294868) INFO:     Application startup complete.
```

The `/v1/models` readiness probe returned successfully:

```json
{
  "id": "minimax-m3",
  "max_model_len": 3072
}
```

## Request Failure Timeline

The first precision test request failed because `max_tokens=8888` exceeded `max_model_len=3072`:

```text
1357:(APIServer pid=2294868) INFO:     127.0.0.1:57800 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
1442:(APIServer pid=2294868) vllm.exceptions.VLLMValidationError: max_tokens=8888 cannot be greater than max_model_len=max_total_tokens=3072.
```

After lowering request `max_tokens`, the request passed validation but crashed during model execution:

```text
1561:(Worker_TP12_EP12 pid=2295939) ERROR 06-24 21:51:00 [multiproc_executor.py:962] IndexError: index 306 is out of bounds for dimension 0 with size 306
5223:(EngineCore pid=2295872) ERROR 06-24 21:51:00 [core.py:1142] EngineCore encountered a fatal error.
5247:(EngineCore pid=2295872) ERROR 06-24 21:51:00 [core.py:1142] RuntimeError: Worker failed with error 'index 306 is out of bounds for dimension 0 with size 306', please check the stack trace above for the root cause
5256:(APIServer pid=2294868) INFO:     127.0.0.1:60426 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```

After the fatal error, the API server shut down and port `11223` refused connections.

## Stack Path

The relevant stack path from the service log:

```text
vllm_ascend/worker/model_runner_v1.py:2260 execute_model
vllm_ascend/models/minimax_m3.py:1349 forward
vllm_ascend/models/minimax_m3.py:855 language_model forward
vllm_ascend/ops/minimax_m3_sparse.py:27 minimax_m3_sparse_forward
vllm_ascend/attention/msa_m3.py:1049 topk_idx = self.indexer(index_query)
vllm_ascend/attention/msa_m3.py:336 score = minimax_m3_index_score_torch(...)
vllm_ascend/attention/msa_m3_ops.py:167 q_heads = idx_q[q_start + t]
IndexError: index 306 is out of bounds for dimension 0 with size 306
```

## Current Analysis

The immediate crash happens in:

```python
# /home/cxy/vllm-ascend-zhx/vllm_ascend/attention/msa_m3_ops.py
q_heads = idx_q[q_start + t]
```

`idx_q` has size `306` on dimension 0, so the valid max index is `305`. Metadata still drives the loop to access index `306`.

The likely mismatch is:

1. `index_query` is sliced by real token count:

```python
# /home/cxy/vllm-ascend-zhx/vllm_ascend/attention/msa_m3.py
num_tokens = index_md.num_actual_tokens
iq = index_query[:num_tokens].view(-1, self.num_index_heads, self.index_head_dim)
```

2. `cu_seqlens_q` is built from `query_start_loc`:

```python
cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(torch.int32)
```

3. In `model_runner_v1.py`, `query_start_loc` can be padded for FIA/cudagraph:

```python
# /home/cxy/vllm-ascend-zhx/vllm_ascend/worker/model_runner_v1.py
num_reqs_padded = self._pad_query_start_loc_for_fia(...)
```

4. `_build_attention_metadata()` passes `num_actual_tokens=num_tokens_unpadded`, while `query_start_loc` may already be padded.

This means M3 sparse/indexer metadata can describe more query positions than the real `index_query` tensor contains. The dummy padded query/request then causes the out-of-bounds access.

## Proposed Fix

Fix this in MiniMax M3 sparse/indexer metadata construction, not by skipping multimodal profiling or changing unrelated runner behavior.

Recommended local fix:

Clamp the M3 `cu_seqlens_q` upper bound to the real prefill token count:

```text
real_prefill_tokens = num_actual_tokens - num_decode_tokens
cu_seqlens_q = clamp(query_start_loc[num_decodes:] - num_decode_tokens, max=real_prefill_tokens)
```

Apply the same logic in both M3 metadata builders:

```text
/home/cxy/vllm-ascend-zhx/vllm_ascend/attention/msa_m3.py
  AscendMiniMaxM3IndexerMetadataBuilder.build()
  AscendMiniMaxM3SparseMetadataBuilder.build()
```

Expected effect:

```text
padded dummy request/query becomes zero-length for M3 sparse/indexer metadata,
so minimax_m3_index_score_torch will not index beyond idx_q.shape[0].
```

After this fix, restart service and rerun:

```bash
cd /home/cxy/minimax_m3
bash start_service_bf16.sh
bash curl_with_throughput.sh
```

Validation target:

```text
1. Service starts normally.
2. Precision script answers C.
3. Output includes completion throughput.
```
