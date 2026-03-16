"""
Serve 命令 - API 服务管理
"""

import click
import subprocess
import os
import signal
import sys
import time
from pathlib import Path
from rich.console import Console

console = Console()


def _get_pid_file():
    """获取 PID 文件路径"""
    cache_dir = Path.home() / ".cache" / "mpa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "server.pid"


def _is_server_running():
    """检查服务是否正在运行"""
    pid_file = _get_pid_file()
    if not pid_file.exists():
        return False, None

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # 检查进程是否存在
        return True, pid
    except (ProcessLookupError, ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return False, None


@click.group()
def serve():
    """API 服务管理命令"""
    pass


@serve.command()
@click.option('--host', default='127.0.0.1', help='绑定地址')
@click.option('--port', default=8000, type=int, help='端口')
@click.option('--workers', type=int, help='Worker 数量（默认 auto）')
@click.option('--daemon', is_flag=True, help='后台运行')
@click.option('--log-level', default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error']),
              help='日志级别')
@click.option('--reload', is_flag=True, help='开发模式：代码变更自动重载')
@click.option('--broker', help='Celery broker URL (启用分布式模式)')
def start(host, port, workers, daemon, log_level, reload, broker):
    """启动 API 服务"""

    # 检查是否已在运行
    running, pid = _is_server_running()
    if running:
        console.print(f"[yellow]服务已在运行 (PID: {pid})[/yellow]")
        console.print(f"访问: http://{host}:{port}")
        console.print(f"文档: http://{host}:{port}/docs")
        return

    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    console.print("[blue]启动 MPA API 服务...[/blue]")
    console.print(f"  地址: {host}:{port}")
    console.print(f"  日志级别: {log_level}")
    if broker:
        console.print(f"  消息队列: {broker}")
    if workers:
        console.print(f"  Workers: {workers}")
    if daemon:
        console.print(f"  后台模式: 是")

    # 构建启动命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        "missionplanalgo.server.app:app",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level,
    ]

    if workers:
        cmd.extend(["--workers", str(workers)])

    if reload:
        cmd.append("--reload")

    # 设置环境变量
    env = os.environ.copy()
    if broker:
        env["MPA_BROKER_URL"] = broker

    try:
        if daemon:
            # 后台模式
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )

            # 保存 PID
            pid_file = _get_pid_file()
            pid_file.write_text(str(process.pid))

            console.print(f"[green]服务已在后台启动 (PID: {process.pid})[/green]")
            console.print(f"访问: http://{host}:{port}")
            console.print(f"文档: http://{host}:{port}/docs")
            console.print(f"查看状态: mpa serve status")
            console.print(f"停止服务: mpa serve stop")
        else:
            # 前台模式
            console.print(f"\n[green]服务已启动！[/green]")
            console.print(f"访问: http://{host}:{port}")
            console.print(f"API 文档: http://{host}:{port}/docs")
            console.print(f"按 Ctrl+C 停止服务\n")

            # 运行服务
            subprocess.run(cmd, env=env)

    except KeyboardInterrupt:
        console.print("\n[yellow]服务已停止[/yellow]")
    except Exception as e:
        console.print(f"[red]启动失败: {e}[/red]")
        raise click.Abort()


@serve.command()
@click.option('--force', is_flag=True, help='强制停止')
@click.option('--timeout', default=30, type=int, help='优雅关闭超时（秒）')
def stop(force, timeout):
    """停止 API 服务"""

    running, pid = _is_server_running()

    if not running:
        console.print("[yellow]服务未在运行[/yellow]")
        return

    console.print(f"[blue]正在停止服务 (PID: {pid})...[/blue]")

    try:
        if force:
            os.kill(pid, signal.SIGKILL)
            console.print("[green]服务已强制停止[/green]")
        else:
            os.kill(pid, signal.SIGTERM)

            # 等待进程退出
            for _ in range(timeout):
                try:
                    os.kill(pid, 0)
                    time.sleep(1)
                except ProcessLookupError:
                    console.print("[green]服务已停止[/green]")
                    break
            else:
                console.print("[yellow]优雅关闭超时，强制停止...[/yellow]")
                os.kill(pid, signal.SIGKILL)

        # 清理 PID 文件
        pid_file = _get_pid_file()
        pid_file.unlink(missing_ok=True)

    except ProcessLookupError:
        console.print("[yellow]服务已经停止[/yellow]")
        pid_file = _get_pid_file()
        pid_file.unlink(missing_ok=True)
    except Exception as e:
        console.print(f"[red]停止失败: {e}[/red]")
        raise click.Abort()


@serve.command()
def status():
    """查看服务状态"""

    running, pid = _is_server_running()

    if not running:
        console.print("[yellow]服务状态: 已停止[/yellow]")
        return

    console.print("[green]服务状态: 运行中[/green]")
    console.print(f"  PID: {pid}")

    # 尝试获取更多信息
    try:
        import psutil
        process = psutil.Process(pid)
        console.print(f"  启动时间: {process.create_time()}")
        console.print(f"  内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        console.print(f"  CPU使用: {process.cpu_percent()}%")
    except ImportError:
        pass

    # 读取配置文件获取端口
    try:
        from missionplanalgo.config import load_config
        config = load_config()
        host = config.get("server", {}).get("host", "127.0.0.1")
        port = config.get("server", {}).get("port", 8000)
        console.print(f"  访问地址: http://{host}:{port}")
        console.print(f"  API 文档: http://{host}:{port}/docs")
    except:
        console.print(f"  默认访问: http://127.0.0.1:8000")


@serve.command()
def reload():
    """重载服务配置"""
    console.print("[blue]重载服务配置...[/blue]")

    running, pid = _is_server_running()
    if not running:
        console.print("[yellow]服务未在运行，无法重载[/yellow]")
        return

    # 发送 SIGHUP 信号触发重载
    try:
        os.kill(pid, signal.SIGHUP)
        console.print("[green]配置已重载[/green]")
    except Exception as e:
        console.print(f"[red]重载失败: {e}[/red]")
