# Qwen3 MoE accuracy regression on the v0.27.1 baseline

This report documents a fix validated with vLLM v0.27.1 and vLLM Ascend
`1ba7ee4e0319e5a0333ed0e50042b91823f49c26`. The supplied 32-question
gsm8k-lite test set scored **100% (32/32)** after the fix, exceeding the
requested 86% threshold. This is a result for that small test set, not the
full GSM8K benchmark or a claim about current upstream branches.

## Root cause

Commit `3eb8dede168ad9e24e3bb548ed35db7e658dd40e` (PR #13158) added a
Python `output_is_reduced` condition around the runtime reduction operator
in `AscendMoERunner._maybe_reduce_final_output`.

In this configuration, profiling first captures an MC2 execution at its
128-token capacity. MC2 has already reduced the expert output, so the
Python condition removes `maybe_all_reduce_tensor_model_parallel` from the
captured graph. The compiled range covers 1–2048 tokens and is subsequently
reused for AllGather prefill. That communication mode needs a TP sum, but
the captured graph no longer contains the operator that performs it.

The vLLM piecewise backend invokes the captured graph directly. A test
that only calls the `torch.compile` wrapper can miss this bug: Dynamo can
recompile when the context changes, whereas direct graph dispatch reuses
the original graph.

Evidence supporting the diagnosis:

- A minimal reproduction calls the actual Ascend reduction method and
  runtime custom-op implementation with a CPU implementation of TP sum.
  After MC2 capture, switching to AllGather produces 16 eagerly but 2 from
  the captured graph. The mocked TP size is 8.
- Replacing only that method with its implementation before #13158 makes
  the same reproduction pass: both outputs are 16.
- The captured NPU computation graph at bisection commit `6b424bfa9` lacks
  `maybe_all_reduce_tensor_model_parallel`. The fixed graph contains it.
- Before the fix, one of the two graph-capture regression cases fails.
  With the fix, both pass, along with the early-reduction cases and the
  existing fused-MoE unit tests.

## Fix and scope

For runners without shared experts or a routed output transform, always
retain the existing runtime reduction custom op in the graph. Its
implementation decides at execution time whether the active communication
mode requires a TP sum. This covers the tested Qwen3-235B-A22B path.

For runners with shared experts or an output transform, preserve the
existing explicit early-reduction behavior. Regression tests cover that
behavior and output truncation. The fix adds no environment variables or
device-to-host synchronization.

This patch targets the historical baseline above. Current upstream `main`
and `releases/v0.27.1rc` have subsequently changed the reduction contract
and removed this custom op; applying this patch to those branches requires
a separate analysis and validation.

## Accuracy validation

The known-good and known-bad endpoints were supplied as already tested.
They were not rerun. There were 28 commits in the inspected interval; the
hardware bisection run used `6b424bfa9`, followed by the targeted graph
reproduction to isolate #13158.

| Ascend revision | Source of result | Correct / total | Accuracy |
| --- | --- | --- | --- |
| `bf7382b5e` (#14249) | Previously verified good by reporter | Not supplied | Not supplied |
| `1ba7ee4e031` (#14142) | Previously verified bad by reporter | Not supplied | Not supplied |
| `6b424bfa9e891d78544c2fd737a5486ddc5f1e35` (#12923) | Two-node run during diagnosis | 0/32 | 0% |
| `1ba7ee4e031` plus this fix | Two-node validation | 32/32 | 100% |

Both measured runs had zero failed requests. The final score was checked
against both the AISBench summary CSV and all 32 per-question grading
records. All questions in the supplied test file were evaluated.

| Setting | Value |
| --- | --- |
| Model | Qwen3-235B-A22B |
| Hardware | Two hosts, eight Ascend 910B3 NPUs per host |
| vLLM | v0.27.1, `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Parallelism | TP=8, DP=2, expert parallel enabled |
| Service seed | 1024 |
| Maximum model length | 40960 |
| Maximum batched tokens / sequences | 2048 / 128 |
| Graph mode | FULL_AND_PIECEWISE |
| Prefix caching / chunked prefill | Enabled / enabled |
| Device memory utilization | 0.9 |
| AISBench tag | `v3.1-20260609-master` |
| AISBench commit | `0da56eadb2ac85c31c2540f4f5b69af3ec5717a5` |
| Prompt configuration | `gsm8k_gen_0_shot_cot_chat_prompt`, original answer extraction |
| Sampling temperature | 0.6 |
| Maximum output length / batch size | 7680 / 256 |

The service configuration was taken from
`tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-A22B-A2.yaml`,
with local model and dataset paths. AISBench ran in a separate container.
A package-list comparison in the evaluation host's inference container
showed that only the intended `vllm_ascend` version changed.

The kernels were rebuilt during diagnosis. The bisection revision and
the bad baseline have identical `csrc`, `CMakeLists.txt`, and `setup.py`,
so subsequent Python-only installations reused those kernels. The fixed
computation graphs were recompiled. Both inference services were stopped
after validation, and all 16 NPUs were confirmed idle.

Inputs are identified by SHA-256 so the small test set cannot be confused
with another GSM8K subset:

| Input | SHA-256 |
| --- | --- |
| Supplied `test.jsonl` (32 questions) | `6b1b25e3c772d87e36c01839e3b91e910793caae412785024700e2c270176bb9` |
| Model `config.json` | `0ecd5d6fe6f2db6739e4e36ab06b88ebe7bd013ef31b9583f43796059a2b23a4` |
| Original case YAML | `ac3930c2c4896947dcef6f1bf3b00b5b92cfc6dab10594b7d3432e5c17938238` |

## Regression tests and checks

In the configured Ascend inference environment:

```bash
pytest -q tests/ut/ops/test_moe_reduction_compile.py tests/ut/ops/test_fused_moe.py
```

Result: **63 passed**, comprising six new regression cases and 57 existing
cases. The graph test captures under either MC2 or AllGather, then directly
executes the captured graph under MC2, AllGather, and MC2 again, comparing
each result with eager execution.

Ruff check, Ruff format, codespell, typos, and whitespace checks passed.
The full `bash format.sh ci` command was attempted on Windows; it did not
fully pass because the gitleaks and logger hook launchers require
`/bin/bash`, and shellcheck was unavailable. Running the logger check
directly through Git Bash passed. The full repository test suite across
all supported models and hardware was not run.

The private validation archive retains the service logs, compiled graphs,
AISBench summaries and per-question results, package snapshots, input
manifest, and rerun scripts. Host addresses, credentials, local mount
paths, and model or dataset contents are intentionally excluded from this
public report.
