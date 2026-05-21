import pytest
from unittest.mock import MagicMock
from pathlib import Path


def test_extract_regulation_core():
    from longhorizon_bench.pipeline.bg_generator import extract_regulation_core
    regs = [
        {"title": "安全生产法", "text": "第三十四条 压力容器定期检验周期为3年，检验期间应降负荷运行。第三十五条 危险化学品储存间距不小于15米。"},
        {"title": "消防法", "text": "第二十条 消防通道宽度不小于4米。"},
    ]
    core = extract_regulation_core(regs, max_chars=5000)
    assert "压力容器" in core
    assert "消防" in core
    assert len(core) <= 5000


def test_generate_background():
    from longhorizon_bench.pipeline.bg_generator import generate_background
    mock_client = MagicMock()
    mock_client.chat.return_value = "华安化工有限公司成立于2005年..." + "扩展内容" * 1000
    chain = {
        "scenario_id": "IND-001",
        "role": "化工厂安全生产管理员",
        "events": [{"id": "E01", "trigger": "压力容器巡检"}],
    }
    reg_core = "压力容器定期检验周期为3年。"
    bg_text = generate_background(mock_client, chain, reg_core)
    assert len(bg_text) > 0
    mock_client.chat.assert_called_once()


def test_save_background_docs(tmp_path):
    from longhorizon_bench.pipeline.bg_generator import save_background_docs
    save_background_docs("IND-001", "这是背景文档内容", tmp_path)
    doc_path = tmp_path / "background_docs" / "IND-001" / "background.txt"
    assert doc_path.exists()
    assert "背景文档" in doc_path.read_text(encoding="utf-8")
