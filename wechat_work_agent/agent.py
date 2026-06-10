"""
Core agent loop — reusable by both the orchestrator and sub-agents.
"""

import asyncio
from typing import Any, Callable

import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


async def run_agent_loop(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    initial_message: str,
    tools: list,
    tool_executor: Callable,
    agent_name: str = "Agent",
    max_turns: int = 20,
    verbose: bool = True,
) -> str:
    """
    Run an agentic loop: send message → execute tool calls → repeat until
    the model stops with end_turn or max_turns is reached.

    Returns the final text response.
    """
    messages = [{"role": "user", "content": initial_message}]

    for turn in range(max_turns):
        if verbose:
            console.print(f"[dim]  [{agent_name}] thinking... (turn {turn + 1})[/dim]")

        response = await client.messages.create(
            model=model,
            max_tokens=8096,
            system=system,
            messages=messages,
            tools=tools,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Collect any text blocks
        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        text_output = "\n".join(text_parts).strip()

        if verbose and text_output:
            console.print(
                Panel(
                    Markdown(text_output),
                    title=f"[cyan]{agent_name}[/cyan]",
                    border_style="cyan",
                    expand=False,
                )
            )

        # Done?
        if response.stop_reason == "end_turn":
            return text_output

        # Execute tool calls in parallel
        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            if verbose:
                names = ", ".join(b.name for b in tool_blocks)
                console.print(f"[yellow]  [{agent_name}] → tools: {names}[/yellow]")

            async def _run_one(block: Any):
                try:
                    result = await tool_executor(block.name, block.input)
                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _serialize(result),
                    }
                except Exception as exc:
                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool error: {exc}",
                        "is_error": True,
                    }

            tool_results = await asyncio.gather(*[_run_one(b) for b in tool_blocks])
            messages.append({"role": "user", "content": list(tool_results)})

    return text_output or "Agent reached maximum turns without a final response."


def _serialize(value: Any) -> str:
    """Convert any tool return value to a string for the API."""
    if isinstance(value, str):
        return value
    try:
        import json
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)
