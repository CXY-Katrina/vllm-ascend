# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.forward_context import override_forward_context

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops import register_custom_ops
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

TP_SIZE = 8


class RoutedOutputReduction(torch.nn.Module):
    """Exercise the runner's reduction boundary without loading experts."""

    _fused_output_is_reduced = AscendMoERunner._fused_output_is_reduced
    _maybe_reduce_final_output = AscendMoERunner._maybe_reduce_final_output

    def __init__(self):
        super().__init__()
        self.ascend_shared_experts = None
        self.routed_output_transform = None

    def forward(self, states):
        return self._maybe_reduce_final_output(states + 1, None, self._fused_output_is_reduced)


@pytest.mark.parametrize("capture_mode", [MoECommType.MC2, MoECommType.ALLGATHER])
def test_captured_moe_reduction_follows_runtime_communication(monkeypatch, capture_mode):
    # Model the TP sum while executing the real runtime communication switch.
    monkeypatch.setattr(register_custom_ops, "tensor_model_parallel_all_reduce", lambda states: states * TP_SIZE)
    captured_graphs = []

    def backend(graph, example_inputs):
        captured_graphs.append(graph)
        return graph.forward

    def context(mode):
        values = {"moe_comm_type": mode, "flash_comm_v1_enabled": False}
        return SimpleNamespace(**values, additional_kwargs=values)

    torch._dynamo.reset()
    library = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        library.impl(
            "maybe_all_reduce_tensor_model_parallel",
            register_custom_ops._maybe_all_reduce_tensor_model_parallel_impl,
        )
        runner = RoutedOutputReduction()
        states = torch.ones(4, 8)
        compiled = torch.compile(runner, backend=backend, fullgraph=True, dynamic=False)
        with override_forward_context(context(capture_mode)):
            compiled(states)

        assert len(captured_graphs) == 1
        # vLLM dispatches the captured graph directly. Calling the Dynamo
        # wrapper here would recompile on context changes and hide the bug.
        graph = captured_graphs[0]
        for mode in (MoECommType.MC2, MoECommType.ALLGATHER, MoECommType.MC2):
            with override_forward_context(context(mode)):
                expected = runner(states)
                actual = graph(states)[0]
            torch.testing.assert_close(actual, expected)
    finally:
        library._destroy()
        torch._dynamo.reset()


@pytest.mark.parametrize("shared_experts,output_transform", [(True, False), (False, True)])
@pytest.mark.parametrize("output_is_reduced", [True, False])
def test_explicit_early_reduction_is_preserved(monkeypatch, shared_experts, output_transform, output_is_reduced):
    runner = RoutedOutputReduction()
    runner.ascend_shared_experts = object() if shared_experts else None
    runner.routed_output_transform = object() if output_transform else None
    reductions = []

    def reduce(states):
        reductions.append(states)
        return states * TP_SIZE

    monkeypatch.setattr(torch.ops.vllm, "maybe_all_reduce_tensor_model_parallel", reduce)
    states = torch.ones(4, 8)
    result = runner._maybe_reduce_final_output(states, 3, output_is_reduced)
    expected = states[:, :3] if output_is_reduced else states[:, :3] * TP_SIZE
    torch.testing.assert_close(result, expected)
    assert len(reductions) == int(not output_is_reduced)
