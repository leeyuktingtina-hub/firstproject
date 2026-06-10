"""
Orchestrator agent — decomposes tasks and spawns parallel sub-agents.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import run_agent_loop
from config import MAX_SUBAGENTS, MODEL, OUTPUT_DIR
from tools import TOOL_DEFINITIONS, build_tool_executor

console = Console()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """You are an intelligent AI work assistant — a powerful agent that helps users complete complex research, analysis, and productivity tasks. Think of yourself as Kimi Work: capable, autonomous, and efficient.

## Core Capabilities
- **Parallel execution**: Use `spawn_subagents` to run multiple research tracks simultaneously
- **Web research**: Search and fetch information from the internet
- **Local file access**: Read, write, and analyze files on the user's computer
- **Data analysis**: Execute Python for calculations, data processing, chart generation
- **Report generation**: Produce structured Excel/Markdown reports

## How to Handle Tasks

**Simple tasks** (single topic, one step): Handle directly using tools. No sub-agents needed.

**Complex tasks** (multiple topics, parallel research, multi-step):
1. Briefly state your plan
2. Call `spawn_subagents` with specific, well-defined tasks for each agent
3. Wait for all results to return
4. Synthesize everything into a clear, comprehensive final output
5. Save important outputs (reports, Excel files) to disk and mention the file paths

## Sub-agent task writing guidelines
- Be specific: "Search for NVIDIA Q4 2025 earnings and extract revenue, gross margin, EPS"
- Include data sources when relevant: "Search 同花顺 and Yahoo Finance for..."
- Specify output format: "Return a structured summary with: company name, score 0-10, key metrics"

## Output format
- Use Markdown for structure (headers, bullet points, tables)
- Summarize key findings prominently
- Always cite data sources
- Save detailed reports to files, show the path
- Be direct — give the user actionable insights

You are running as the orchestrator. Sub-agents cannot spawn further sub-agents.
"""

SUBAGENT_SYSTEM = """You are a focused research sub-agent executing one specific task. Be thorough, use your tools, and return a clear, structured result.

## Available tools
- `web_search` — search the internet
- `fetch_url` — fetch full content of a URL
- `read_file` — read local files
- `list_files` — list files in a directory
- `execute_python` — run Python code (pandas/openpyxl pre-imported)
- `write_file` — save results to a file
- `create_excel_report` — create Excel workbooks

## Instructions
1. Execute the assigned task as thoroughly as possible
2. Use multiple searches and URL fetches to cross-verify information
3. Cite your sources (URLs)
4. Return a structured result — use headers and bullet points
5. If you create files, mention the full path

Focus on quality and accuracy. Be concise but complete.
"""

# ---------------------------------------------------------------------------
# spawn_subagents tool definition
# ---------------------------------------------------------------------------

SPAWN_SUBAGENTS_TOOL = {
    "name": "spawn_subagents",
    "description": (
        "Spawn multiple sub-agents to execute independent tasks IN PARALLEL. "
        "Each agent runs its own agentic loop with full tool access. "
        "Use this when a task can be split into independent research tracks. "
        f"Maximum {MAX_SUBAGENTS} agents per call. Returns a dict of {{agent_name: result}}."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": f"List of tasks to execute in parallel (max {MAX_SUBAGENTS})",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short label for this agent (e.g. 'nvidia_financials', 'market_news')",
                        },
                        "task": {
                            "type": "string",
                            "description": "Full, specific task description for this sub-agent",
                        },
                    },
                    "required": ["name", "task"],
                },
            }
        },
        "required": ["tasks"],
    },
}


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = MODEL
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    async def run(self, task: str) -> str:
        """Run the orchestrator on a user task."""
        console.print()
        console.rule(f"[bold cyan]New Task[/bold cyan]")
        console.print(f"[bold]{task}[/bold]\n")

        # Build orchestrator tools: all base tools + spawn_subagents
        orch_tools = TOOL_DEFINITIONS + [SPAWN_SUBAGENTS_TOOL]
        base_executor = build_tool_executor()

        async def orch_executor(name: str, inputs: dict) -> Any:
            if name == "spawn_subagents":
                return await self._spawn_subagents(inputs.get("tasks", []))
            return await base_executor(name, inputs)

        result = await run_agent_loop(
            client=self.client,
            model=self.model,
            system=ORCHESTRATOR_SYSTEM,
            initial_message=task,
            tools=orch_tools,
            tool_executor=orch_executor,
            agent_name="Orchestrator",
            max_turns=25,
            verbose=True,
        )

        console.rule("[bold green]Task Complete[/bold green]")
        return result

    async def _spawn_subagents(self, tasks: list) -> dict:
        """Spawn N sub-agents in parallel and return their results."""
        # Cap to max allowed
        tasks = tasks[:MAX_SUBAGENTS]

        if not tasks:
            return {"error": "No tasks provided to spawn_subagents"}

        # Display what we're spawning
        table = Table(title=f"[yellow]Spawning {len(tasks)} Sub-Agents in Parallel[/yellow]",
                      border_style="yellow", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Agent Name", style="cyan", width=20)
        table.add_column("Task")
        for i, t in enumerate(tasks, 1):
            task_preview = t["task"][:100] + ("..." if len(t["task"]) > 100 else "")
            table.add_row(str(i), t["name"], task_preview)
        console.print(table)

        # Run all sub-agents in parallel
        async def run_one(task_item: dict) -> tuple[str, str]:
            agent_name = task_item["name"]
            sub_executor = build_tool_executor()  # fresh executor per agent
            try:
                result = await run_agent_loop(
                    client=self.client,
                    model=self.model,
                    system=SUBAGENT_SYSTEM,
                    initial_message=task_item["task"],
                    tools=TOOL_DEFINITIONS,  # no spawn_subagents for sub-agents
                    tool_executor=sub_executor,
                    agent_name=f"Sub[{agent_name}]",
                    max_turns=15,
                    verbose=True,
                )
                return agent_name, result
            except Exception as exc:
                console.print(f"[red]Sub-agent [{agent_name}] failed: {exc}[/red]")
                return agent_name, f"Error: {exc}"

        console.print(f"\n[yellow]Running {len(tasks)} agents in parallel...[/yellow]\n")
        pairs = await asyncio.gather(*[run_one(t) for t in tasks])
        results = dict(pairs)

        console.print(f"\n[green]✓ All {len(tasks)} sub-agents completed.[/green]\n")
        return results
