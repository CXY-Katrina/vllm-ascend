"""NPU precision and no-host-read regressions for the installed MiniMax path."""

import math
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionImpl as MiniMaxDSparkFlashImpl,
)
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionMetadataBuilder as MiniMaxDSparkFlashMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.DEVICE_LENGTH_FLASH_ATTN),
    reason="Requires the A5 CANN FlashAttn stack",
)


class NoDeviceHostRead(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.cpu_scalar_reads = 0
        self.npu_scalar_reads = 0
        self.device_to_host_copies = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device_inputs = [
            x for x in tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor) and x.device.type == "npu"
        ]
        if device_inputs:
            if "_local_scalar_dense" in str(func):
                self.npu_scalar_reads += 1
            assert "_local_scalar_dense" not in str(func), f"NPU scalar synchronization: {func}"
            if "_to_copy" in str(func) or "copy_" in str(func):
                destination = kwargs.get("device")
                if destination is not None:
                    if torch.device(destination).type == "cpu":
                        self.device_to_host_copies += 1
                    assert torch.device(destination).type != "cpu", f"NPU to CPU: {func}"
                if "copy_" in str(func) and isinstance(args[0], torch.Tensor):
                    if args[0].device.type == "cpu":
                        self.device_to_host_copies += 1
                    assert args[0].device.type != "cpu", f"NPU to CPU copy: {func}"
        elif "_local_scalar_dense" in str(func):
            self.cpu_scalar_reads += 1
        return func(*args, **kwargs)


def config_for_minimax(architecture="MiniMaxM31FlashA3DSparkModel"):
    target = NS(hf_text_config=NS(model_type="minimax_m3_text"), hf_config=NS(model_type="minimax_m3"))
    return NS(
        speculative_config=NS(
            method="dspark",
            enforce_eager=True,
            enable_adaptive_verification=False,
            target_model_config=target,
            draft_model_config=NS(architectures=[architecture]),
        ),
        model_config=target,
        parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=NS(cache_dtype="bfloat16"),
        quant_config=None,
    )


def selects_flash_attention(config, kv_dtype="bfloat16", model_tag="dspark_head"):
    from vllm.v1.attention.backend import AttentionType

    from vllm_ascend.platform import _validate_flash_attn_backend

    selector = NS(
        attn_type=AttentionType.DECODER,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        use_pcp=False,
        use_dcp=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        use_batch_invariant=False,
        use_adaptive_verification=False,
    )
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch("vllm.compilation.backends.model_tag", model_tag),
    ):
        return _validate_flash_attn_backend((False, False, False), selector)


@pytest.mark.parametrize("architecture", ["MiniMaxM31FlashA3DSparkModel", "DSparkMiniMaxDraftModel"])
def test_native_and_prepared_minimax_drafts_select_flash_attention(architecture):
    assert selects_flash_attention(config_for_minimax(architecture))


@pytest.mark.parametrize("architecture", ["MiniMaxM31FlashA3DSparkModel", "DSparkMiniMaxDraftModel"])
@pytest.mark.parametrize("model_tag", ["backbone", "default", "dflash_head"])
def test_target_and_other_model_tags_keep_their_existing_backend(model_tag, architecture):
    assert not selects_flash_attention(config_for_minimax(architecture), model_tag=model_tag)


def test_unrelated_draft_keeps_its_existing_backend():
    assert not selects_flash_attention(config_for_minimax("Qwen3DSparkModel"))


@pytest.mark.parametrize(
    "field,value", [("method", "mtp"), ("enforce_eager", False), ("enable_adaptive_verification", True)]
)
def test_other_speculative_paths_keep_their_existing_backend(field, value):
    config = config_for_minimax()
    assert selects_flash_attention(config)
    setattr(config.speculative_config, field, value)
    assert not selects_flash_attention(config)


@pytest.mark.parametrize(
    "parallel_field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_context_and_pipeline_parallel_keep_host_sync(parallel_field):
    config = config_for_minimax()
    setattr(config.parallel_config, parallel_field, 2)
    assert not selects_flash_attention(config)


def test_flash_selection_checks_each_layers_actual_kv_dtype():
    config = config_for_minimax()
    assert selects_flash_attention(config, "bfloat16")
    assert not selects_flash_attention(config, "fp8")


def test_full_decode_graph_replay_refreshes_rejected_device_lengths():
    torch.manual_seed(18717)
    q = torch.randn(16, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(4, 128, 1, 128, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    query, key, value = q.npu(), k.npu(), v.npu()
    output = torch.empty_like(query)
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads, impl.sinks = 16, 1, None
    impl.scale, impl.sliding_window, impl.use_bnsd_kv_cache = 1 / math.sqrt(128), 4096, False
    impl.key_cache, impl.value_cache = key, value
    common = make_common([8, 8], [129, 63])
    builder = make_builder()
    for _ in range(3):
        impl.forward_impl(query, None, None, (key, value), builder.build(0, common), output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    metadata = builder.build(0, common)
    with torch.npu.graph(graph):
        impl.forward_impl(query, None, None, (key, value), metadata, output)
    graph.replay()
    expected = reference(q, k, v, [8, 8], [129, 63], True, 4096)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)
    common.seq_lens.copy_(torch.tensor([73, 197], dtype=torch.int32, device="npu"))
    with NoDeviceHostRead():
        graph.replay()
    expected = reference(q, k, v, [8, 8], [73, 197], True, 4096)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


def make_common(q_lengths, kv_lengths, device="npu"):
    qsl = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32)
    n = len(q_lengths)
    return NS(
        num_actual_tokens=sum(q_lengths),
        num_reqs=n,
        max_query_len=max(q_lengths),
        max_seq_len=256,
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        # Deliberately wrong CPU lengths must never reach an attention kernel.
        seq_lens_cpu=torch.full((n,), 256, dtype=torch.int32),
        seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        block_table_tensor=torch.arange(n * 2, dtype=torch.int32, device=device).view(n, 2),
        slot_mapping=torch.arange(sum(q_lengths), dtype=torch.int64, device=device),
        causal=True,
        attn_state=AscendAttentionState.SpecDecoding,
        context_parallel_metadata=None,
    )


def make_builder():
    builder = object.__new__(MiniMaxDSparkFlashMetadataBuilder)
    builder.model_config = NS(runner_type="generate")
    builder.attn_mask_builder = AttentionMaskBuilder(torch.device("npu"))
    builder.attn_mask_builder.get_splitfuse_attn_mask()
    return builder


def reference(query, key, value, q_lengths, kv_lengths, causal, window):
    outputs = []
    q_offset = 0
    for i, (ql, kl) in enumerate(zip(q_lengths, kv_lengths)):
        if ql == 0:
            continue
        q = query[q_offset : q_offset + ql].float().transpose(0, 1)
        k = (
            key[i * 2 : i * 2 + 2]
            .reshape(256, 1, 128)[:kl]
            .float()
            .repeat_interleave(query.shape[1], dim=1)
            .transpose(0, 1)
        )
        v = (
            value[i * 2 : i * 2 + 2]
            .reshape(256, 1, 128)[:kl]
            .float()
            .repeat_interleave(query.shape[1], dim=1)
            .transpose(0, 1)
        )
        scores = q @ k.transpose(1, 2) / math.sqrt(128)
        qpos = torch.arange(kl - ql, kl)[:, None]
        kpos = torch.arange(kl)[None, :]
        keep = torch.ones((ql, kl), dtype=torch.bool)
        if causal:
            keep &= kpos <= qpos
        if window is not None:
            keep &= kpos >= qpos - (window - 1)
            if not causal:
                keep &= kpos <= qpos + (window - 1)
        scores.masked_fill_(~keep[None], float("-inf"))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
        q_offset += ql
    return torch.cat(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal,window", [(True, None), (False, None), (True, 16), (False, 16), (False, 4096)])
@pytest.mark.parametrize("layout", ["PA_BBND", "PA_BNBD"])
@pytest.mark.parametrize("width", [7, 8])
def test_flash_attention_matches_reference_after_rejection_and_padding(dtype, causal, window, layout, width):
    torch.manual_seed(18717)
    q_lengths, kv_lengths = [width, width, 0], [129, 63, 0]
    q_cpu = torch.randn(width * 2, 16, 128, dtype=dtype)
    k_cpu = torch.randn(6, 128, 1, 128, dtype=dtype)
    v_cpu = torch.randn_like(k_cpu)
    query, key, value = q_cpu.npu(), k_cpu.npu(), v_cpu.npu()
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads = 16, 1
    impl.scale, impl.sliding_window, impl.sinks = 1 / math.sqrt(128), window, None
    impl.use_bnsd_kv_cache = layout == "PA_BNBD"
    if impl.use_bnsd_kv_cache:
        key, value = key.permute(0, 2, 1, 3).contiguous(), value.permute(0, 2, 1, 3).contiguous()
    impl.key_cache, impl.value_cache = key, value
    common = make_common(q_lengths, kv_lengths)
    common.causal = causal
    builder = make_builder()
    metadata = builder.build(0, common)
    output = torch.empty_like(query)
    with NoDeviceHostRead():
        impl.forward_impl(query, None, None, (key, value), metadata, output)
        first_plan = next(iter(metadata.flash_attn_plans.values()))[0]
        impl.forward_impl(query, None, None, (key, value), metadata, output)
        assert first_plan is next(iter(metadata.flash_attn_plans.values()))[0]
    expected = reference(q_cpu, k_cpu, v_cpu, q_lengths, kv_lengths, causal, window)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


@pytest.mark.parametrize("architecture", ["MiniMaxM31FlashA3DSparkModel", "DSparkMiniMaxDraftModel"])
def test_runner_skips_both_copy_and_wait_but_retains_pp_counts(architecture):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = config_for_minimax(architecture)
    runner.model_config = runner.vllm_config.model_config
    runner.speculative_config = runner.vllm_config.speculative_config
    runner.sync_spec_pp_cpu_counts = False
    runner.attn_groups = [[NS(backend=AscendFlashAttentionBackend)]]
    runner.speculator = object()
    runner.num_computed_tokens_event = Mock()
    runner.req_states = NS(req_id_to_index={"a": 0, "b": 1}, num_computed_tokens_cpu=torch.tensor([100, 200]))
    runner.input_buffers = NS(seq_lens_cpu=torch.zeros(2, dtype=torch.int32))
    scheduled = NS(num_scheduled_tokens={"a": 8, "b": 8}, scheduled_cached_reqs=NS(req_ids=["a", "b"]))
    with NoDeviceHostRead(), patch("torch.cuda.current_stream", side_effect=AssertionError("D2H producer ran")):
        runner._copy_num_computed_tokens_to_cpu()
        runner._update_seq_lens_cpu(scheduled, ["b", "a"])
    runner.num_computed_tokens_event.synchronize.assert_not_called()
    assert runner.input_buffers.seq_lens_cpu.tolist() == [208, 108]
    runner.attn_groups[0][0].backend = AscendAttentionBackend
    assert not runner._can_skip_spec_cpu_length_sync()
    runner.attn_groups[0][0].backend = AscendFlashAttentionBackend
    runner.sync_spec_pp_cpu_counts = True
    assert not runner._can_skip_spec_cpu_length_sync()


@pytest.mark.parametrize("architecture", ["MiniMaxM31FlashA3DSparkModel", "DSparkMiniMaxDraftModel"])
def test_target_host_length_consumer_keeps_shared_runner_sync(architecture):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
    from vllm_ascend.models.minimax_m3.msa_m3 import (
        AscendMiniMaxM3IndexerBackend,
        AscendMiniMaxM3SparseBackend,
    )
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = object.__new__(NPUModelRunner)
    runner.model_config = config_for_minimax().model_config
    runner.speculative_config = config_for_minimax(architecture).speculative_config
    runner.speculator = object()
    runner.sync_spec_pp_cpu_counts = False
    runner.attn_groups = [
        [
            NS(backend=backend)
            for backend in (AscendMiniMaxM3SparseBackend, AscendMiniMaxM3IndexerBackend, AscendFlashAttentionBackend)
        ]
    ]
    assert runner._can_skip_spec_cpu_length_sync()
    runner.attn_groups[0].append(NS(backend=AscendAttentionBackend))
    assert not runner._can_skip_spec_cpu_length_sync()


def test_reordered_rejected_requests_derive_exact_positions_on_npu():
    from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens

    request_order = torch.tensor([1, 0], dtype=torch.int32, device="npu")
    boundaries = torch.tensor([0, 8, 16], dtype=torch.int32, device="npu")
    corrected_counts = torch.tensor([97, 61], dtype=torch.int32, device="npu")
    positions = torch.empty(16, dtype=torch.int64, device="npu")
    seq_lens = torch.empty(2, dtype=torch.int32, device="npu")
    with NoDeviceHostRead():
        prepare_pos_seq_lens(request_order, boundaries, corrected_counts, positions, seq_lens)
    torch.testing.assert_close(seq_lens.cpu(), torch.tensor([69, 105], dtype=torch.int32))
    torch.testing.assert_close(positions.cpu(), torch.cat((torch.arange(61, 69), torch.arange(97, 105))))


def test_fp8_sparse_full_decode_graph_replay_uses_new_lengths_and_indices():
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import minimax_m3_sparse_attn_decode

    torch.manual_seed(18717)
    q = torch.randn(16, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(4, 128, 1, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn(4, 128, 1, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    query, key, value = q.npu(), k.npu(), v.npu()
    common = make_common([8, 8], [129, 63])
    topk = torch.full((1, 16, 16), -1, dtype=torch.int32, device="npu")
    topk[:, :, 0] = 0
    topk[:, :8, 1] = 1
    output = torch.empty_like(query)
    cu_seqlens_q = common.query_start_loc.to(torch.int64)

    def forward():
        minimax_m3_sparse_attn_decode(
            query,
            (key, value),
            topk,
            common.block_table_tensor,
            common.seq_lens,
            1,
            1 / math.sqrt(128),
            output,
            decode_query_len=8,
            cu_seqlens_q=cu_seqlens_q,
        )

    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    common.seq_lens.copy_(torch.tensor([73, 197], dtype=torch.int32, device="npu"))
    topk[:, :8, 1] = -1
    topk[:, 8:, 1] = 1
    with NoDeviceHostRead():
        graph.replay()
    expected = reference(
        q.to(torch.float8_e4m3fn).to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        [8, 8],
        [73, 197],
        True,
        None,
    )
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


def test_133k_context_swa_preserves_device_lengths():
    torch.manual_seed(18717)
    length, block_size, window = 133000, 128, 4096
    blocks = (length + block_size - 1) // block_size
    q = torch.randn(8, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(blocks, block_size, 1, 128, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    query, key, value = q.npu(), k.npu(), v.npu()
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads, impl.sinks = 16, 1, None
    impl.scale, impl.sliding_window, impl.use_bnsd_kv_cache = 1 / math.sqrt(128), window, False
    impl.key_cache, impl.value_cache = key, value
    common = make_common([8], [length])
    common.max_seq_len = length
    common.block_table_tensor = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    builder = make_builder()
    output = torch.empty_like(query)
    with NoDeviceHostRead():
        impl.forward_impl(query, None, None, (key, value), builder.build(0, common), output)
    # The earliest query sees 4096 positions; later queries slide forward.
    start = length - 8 - window + 1
    keys = k.view(-1, 1, 128)[start:length].float().repeat_interleave(16, dim=1).transpose(0, 1)
    values = v.view(-1, 1, 128)[start:length].float().repeat_interleave(16, dim=1).transpose(0, 1)
    scores = q.float().transpose(0, 1) @ keys.transpose(1, 2) / math.sqrt(128)
    qpos = torch.arange(length - 8, length)[:, None]
    kpos = torch.arange(start, length)[None, :]
    keep = (kpos <= qpos) & (kpos >= qpos - window + 1)
    scores.masked_fill_(~keep[None], float("-inf"))
    expected = (scores.softmax(-1) @ values).transpose(0, 1)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


def test_attention_forward_writes_current_draft_kv_before_reading():
    from vllm.v1.attention.backend import AttentionType

    torch.manual_seed(18717)
    q_lengths, kv_lengths = [7, 7], [129, 63]
    q = torch.randn(14, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(4, 128, 1, 128, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    new_k = torch.randn(14, 1, 128, dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    slots = []
    offset = 0
    for index, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        k[index * 2 : index * 2 + 2].view(256, 1, 128)[kv_len - q_len : kv_len] = new_k[offset : offset + q_len]
        v[index * 2 : index * 2 + 2].view(256, 1, 128)[kv_len - q_len : kv_len] = new_v[offset : offset + q_len]
        slots.extend(range(index * 256 + kv_len - q_len, index * 256 + kv_len))
        offset += q_len
    cache_k, cache_v = torch.zeros_like(k).npu(), torch.zeros_like(v).npu()
    # Seed only historical context; the attention entry point must insert Q's KV.
    context_k, context_v = k.clone(), v.clone()
    for slot in slots:
        context_k.view(-1, 1, 128)[slot] = 0
        context_v.view(-1, 1, 128)[slot] = 0
    cache_k.copy_(context_k.npu())
    cache_v.copy_(context_v.npu())
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads, impl.sinks = 16, 1, None
    impl.scale, impl.sliding_window, impl.use_bnsd_kv_cache = 1 / math.sqrt(128), 4096, False
    impl.key_cache, impl.value_cache = cache_k, cache_v
    impl.pcp_enabled, impl.is_kv_producer = False, False
    impl.kv_sharing_target_layer_name, impl.attn_type = None, AttentionType.DECODER
    common = make_common(q_lengths, kv_lengths)
    common.causal = False
    common.slot_mapping = torch.tensor(slots, dtype=torch.int64, device="npu")
    metadata = make_builder().build(0, common)
    layer = NS(layer_name="draft.layers.0.self_attn.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    query, keys, values = q.npu(), new_k.npu(), new_v.npu()
    output = torch.empty_like(query)
    with NoDeviceHostRead():
        impl.forward(layer, query, keys, values, (cache_k, cache_v), metadata, output)
    expected = reference(q, k, v, q_lengths, kv_lengths, False, 4096)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)
