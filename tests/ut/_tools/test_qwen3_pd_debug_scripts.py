# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "examples" / "disaggregated_prefill_v1" / "qwen3_pd_debug"


@pytest.mark.parametrize(
    ("role", "expected_layout", "expected_kv_role", "expected_kv_port"),
    [
        ("prefill", "DP=2, TP=8", "kv_producer", "30000"),
        ("decode", "DP=4, TP=4", "kv_consumer", "30100"),
    ],
)
def test_pd_service_dry_run(role: str, expected_layout: str, expected_kv_role: str, expected_kv_port: str):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "run_pd_service.sh"),
            role,
            "--local-ip",
            "10.0.0.10",
            "--nic-name",
            "eth0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected_layout in result.stdout
    assert expected_kv_role in result.stdout
    assert expected_kv_port in result.stdout
    assert "--data-parallel-address 10.0.0.10" in result.stdout
    assert "--kv-transfer-config" in result.stdout


def test_proxy_dry_run_uses_both_roles():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "run_proxy.sh"),
            "--proxy-ip",
            "10.0.0.10",
            "--prefill-ip",
            "10.0.0.10",
            "--decode-ip",
            "10.0.0.11",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--host 10.0.0.10" in result.stdout
    assert "--prefiller-hosts 10.0.0.10" in result.stdout
    assert "--decoder-hosts 10.0.0.11" in result.stdout
    assert "--prefiller-ports 8080" in result.stdout
    assert "--decoder-ports 8080" in result.stdout
