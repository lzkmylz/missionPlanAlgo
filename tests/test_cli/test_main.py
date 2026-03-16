"""
测试 CLI 主入口
"""

import pytest
from click.testing import CliRunner
from missionplanalgo.cli.main import cli


@pytest.fixture
def runner():
    """CLI 测试运行器"""
    return CliRunner()


def test_cli_help(runner):
    """测试 CLI 帮助"""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Mission Planning Algorithm CLI' in result.output


def test_cli_version(runner):
    """测试 CLI 版本"""
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '1.0.0' in result.output


def test_schedule_help(runner):
    """测试 schedule 命令帮助"""
    result = runner.invoke(cli, ['schedule', '--help'])
    assert result.exit_code == 0
    assert '任务调度命令' in result.output


def test_visibility_help(runner):
    """测试 visibility 命令帮助"""
    result = runner.invoke(cli, ['visibility', '--help'])
    assert result.exit_code == 0
    assert '可见性计算命令' in result.output


def test_serve_help(runner):
    """测试 serve 命令帮助"""
    result = runner.invoke(cli, ['serve', '--help'])
    assert result.exit_code == 0
    assert 'API 服务管理命令' in result.output


def test_config_help(runner):
    """测试 config 命令帮助"""
    result = runner.invoke(cli, ['config', '--help'])
    assert result.exit_code == 0
    assert '配置管理命令' in result.output
