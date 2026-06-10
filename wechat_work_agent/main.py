#!/usr/bin/env python3
"""
WeChat Work Agent — 智能工作助手
A multi-agent AI system inspired by Kimi Work.

Usage:
  python main.py                    # Interactive REPL
  python main.py "your task here"   # Single task mode
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()

from config import ANTHROPIC_API_KEY, MAX_SUBAGENTS, MODEL, OUTPUT_DIR
from orchestrator import Orchestrator

console = Console()

BANNER = r"""
 __        __       ___ _           _      __          __        _
 \ \      / /__ ___/ __| |__   __ _| |_    \ \        / /__  _ __| | __
  \ \ /\ / / _ \ __/ /  | '_ \ / _` | __|    \ \ /\ / / _ \| '__| |/ /
   \ V  V /  __/ |_/ /___| | | | (_| | |_      \ V  V / (_) | |  |   <
    \_/\_/ \___|\__\____|_| |_|\__,_|\__|      \_/\_/ \___/|_|  |_|\_\
                   _                    _
                  /_\  __ _ ___ _ _  __| |_
                 / _ \/ _` / -_) ' \/ _` _|
                /_/ \_\__, \___|_||_\__,\__|
                      |___/
"""

HELP_TEXT = """
## 示例任务

### 金融投研
- `帮我分析英伟达最新财报，输出一份投资研报`
- `搜索A股AI光模块概念股，验证各公司的真实业务，给出含金量评分`
- `帮我查一下近期港股打新情况，哪只值得打`

### 数据分析
- `读取 ./data/sales.csv，分析销售趋势，生成图表`
- `对比分析特斯拉、比亚迪、蔚来2025年的销量数据`

### 信息整理
- `收集今天AI行业最重要的5条新闻，输出结构化摘要`
- `调研国内主流大模型最新动态，对比各家产品能力`

### 办公效率
- `扫描当前目录下的文件，帮我生成工作周报`
- `帮我写一份关于XX主题的市场调研报告`

### 多 Agent 并行示例
- `同时调研：特斯拉财报 + 比亚迪财报 + 蔚来财报，然后做横向对比`

---
输入 `exit` 或 `q` 退出，输入 `help` 查看示例。
"""


async def run_interactive():
    """Interactive REPL mode."""
    console.print(BANNER, style="bold cyan")
    console.print(
        Panel(
            f"[bold]模型:[/bold] {MODEL}  |  "
            f"[bold]最大并行 Agent 数:[/bold] {MAX_SUBAGENTS}  |  "
            f"[bold]输出目录:[/bold] {OUTPUT_DIR}",
            title="[green]系统就绪[/green]",
            border_style="green",
        )
    )
    console.print("[dim]输入 'help' 查看示例任务，输入 'exit' 退出[/dim]\n")

    orchestrator = Orchestrator()

    while True:
        try:
            task = Prompt.ask("\n[bold green]给我一个任务[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]已退出。[/yellow]")
            break

        task = task.strip()
        if not task:
            continue
        if task.lower() in ("exit", "quit", "q", "退出"):
            console.print("[yellow]已退出。[/yellow]")
            break
        if task.lower() in ("help", "帮助", "示例"):
            console.print(Markdown(HELP_TEXT))
            continue

        try:
            await orchestrator.run(task)
        except Exception as exc:
            console.print(f"[red]错误: {exc}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


async def run_single(task: str):
    """Single task mode (non-interactive)."""
    orchestrator = Orchestrator()
    await orchestrator.run(task)


def main():
    if not ANTHROPIC_API_KEY:
        console.print(
            "[red bold]错误: 未找到 ANTHROPIC_API_KEY。[/red bold]\n"
            "请在项目根目录创建 .env 文件并填入:\n"
            "  ANTHROPIC_API_KEY=your_key_here\n\n"
            "参考: wechat_work_agent/.env.example"
        )
        sys.exit(1)

    # If a task is passed as CLI arg, run it once and exit
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        asyncio.run(run_single(task))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
