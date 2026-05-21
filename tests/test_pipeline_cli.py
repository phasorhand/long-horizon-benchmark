# tests/test_pipeline_cli.py
from click.testing import CliRunner


def test_cli_group_exists():
    from longhorizon_bench.pipeline.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.output
    assert "extract" in result.output
    assert "generate" in result.output
    assert "validate" in result.output
    assert "run-all" in result.output
