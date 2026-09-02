from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="qwen3-235b-pd-stream-chat",
        # Local tokenizer/model path visible to the AISBench client.
        path="/path/to/Qwen3-235B-A22B",
        # Must match the model name returned by the PD proxy's /v1/models.
        model="/path/to/Qwen3-235B-A22B",
        stream=True,
        request_rate=11.2,
        use_timestamp=False,
        retry=2,
        api_key="",
        # Send requests to the PD proxy, not the P/D backend port 8080.
        host_ip="<PREFILL_NODE_IP>",
        host_port=8000,
        url="",
        max_out_len=1500,
        batch_size=700,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
