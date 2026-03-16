"""
CLI 主入口

提供 missionplanalgo 的命令行接口。
"""

import click
from rich.console import Console
from rich.traceback import install

# 导入命令模块（这会触发 _compat 中的路径设置）
from missionplanalgo._compat import ensure_project_in_path  # noqa: F401

from .commands import schedule, visibility, serve, config

# 启用 Rich 的漂亮 traceback
install(show_locals=True)

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="mpa")
@click.option('--config', '-c', type=click.Path(), help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.pass_context
def cli(ctx, config, verbose):
    """
    Mission Planning Algorithm CLI

    卫星任务规划算法库命令行工具。

    Examples:
        \b
        # 执行调度
        mpa schedule run -s scenario.json -c cache.json

        # 计算可见性
        mpa visibility compute -s scenario.json

        # 启动 API 服务
        mpa serve start

        # 查看帮助
        mpa --help
        mpa schedule --help
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose

    # 加载配置
    if config:
        from missionplanalgo.config import load_config
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config'] = {}


# 注册子命令
cli.add_command(schedule.schedule)
cli.add_command(visibility.visibility)
cli.add_command(serve.serve)
cli.add_command(config.config_cmd)


def main():
    """CLI 入口点"""
    cli()


if __name__ == '__main__':
    main()
