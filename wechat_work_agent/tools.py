"""
All tool implementations for the WeChat Work Agent.
"""

import asyncio
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------

async def web_search(query: str, max_results: int = 10) -> list[dict]:
    """Search the web using DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS
        # Run sync ddg in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=max_results))
        )
        return results
    except Exception as e:
        return [{"error": str(e), "title": "Search failed", "href": "", "body": ""}]


async def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch and parse a web page, returning clean text content."""
    try:
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "html" in content_type:
                soup = BeautifulSoup(response.text, "lxml")
                # Remove noise elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "ads"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                # Collapse blank lines
                lines = [l for l in text.splitlines() if l.strip()]
                text = "\n".join(lines)
                return text[:max_chars]
            elif "json" in content_type:
                return response.text[:max_chars]
            else:
                return response.text[:max_chars]

    except Exception as e:
        return f"Error fetching {url}: {e}"


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Read a local file and return its content."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"File not found: {path}"
        size = p.stat().st_size
        if size > 2 * 1024 * 1024:  # 2MB limit
            return f"File too large ({size // 1024}KB). Please specify a smaller file."
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


def list_files(directory: str = ".", pattern: str = "**/*") -> list[str]:
    """List files in a directory matching a glob pattern."""
    try:
        p = Path(directory).expanduser().resolve()
        if not p.exists():
            return [f"Directory not found: {directory}"]
        files = sorted(p.glob(pattern))
        # Only return files, not directories, limit to 200
        file_list = [str(f) for f in files if f.is_file()][:200]
        return file_list if file_list else [f"No files found in {directory} matching {pattern}"]
    except Exception as e:
        return [f"Error listing files: {e}"]


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(OUTPUT_DIR) / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content):,} chars to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# Code execution tool
# ---------------------------------------------------------------------------

def execute_python(code: str) -> str:
    """Execute Python code in an isolated namespace and return stdout/result."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf_out
    sys.stderr = buf_err

    # Provide useful imports in the execution namespace
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "json": json,
        "os": os,
        "Path": Path,
        "OUTPUT_DIR": OUTPUT_DIR,
    }

    try:
        # Try to import pandas/openpyxl if available
        try:
            import pandas as pd
            import openpyxl
            namespace["pd"] = pd
            namespace["openpyxl"] = openpyxl
        except ImportError:
            pass

        exec(compile(code, "<agent_code>", "exec"), namespace)
        output = buf_out.getvalue()
        errors = buf_err.getvalue()

        parts = []
        if output:
            parts.append(f"Output:\n{output.rstrip()}")
        if errors:
            parts.append(f"Stderr:\n{errors.rstrip()}")
        if not parts:
            parts.append("Code executed successfully (no output).")
        return "\n".join(parts)[:5000]

    except Exception:
        return f"Execution error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


# ---------------------------------------------------------------------------
# Report generation tool
# ---------------------------------------------------------------------------

def create_excel_report(sheets_data: dict, filename: str) -> str:
    """Create an Excel workbook with multiple sheets from dict data.

    sheets_data: {"SheetName": [{"col1": val, ...}, ...], ...}
    """
    try:
        import pandas as pd

        output_path = Path(OUTPUT_DIR) / filename
        if not filename.endswith(".xlsx"):
            output_path = output_path.with_suffix(".xlsx")

        with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
            for sheet_name, rows in sheets_data.items():
                if isinstance(rows, list) and rows:
                    df = pd.DataFrame(rows)
                elif isinstance(rows, dict):
                    df = pd.DataFrame([rows])
                else:
                    df = pd.DataFrame({"data": [str(rows)]})
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        return f"Excel report saved to: {output_path}"
    except Exception as e:
        return f"Error creating Excel report: {e}"


# ---------------------------------------------------------------------------
# Tool definitions (Claude API format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet. "
            "Use for finding current information, news, financial data, company profiles, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10, max 20)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch and parse a web page, returning its text content. "
            "Use to get detailed content from a specific URL found in search results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a local file and return its contents. Supports text, CSV, JSON, markdown, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory. Use to discover what local files are available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (default: **/*)",
                    "default": "**/*",
                },
            },
        },
    },
    {
        "name": "execute_python",
        "description": (
            "Execute Python code and return stdout output. "
            "pandas (as pd) and openpyxl are pre-imported. "
            "Use for data analysis, calculations, creating charts, processing data, generating reports. "
            f"Save output files to OUTPUT_DIR which is pre-set to '{OUTPUT_DIR}'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "write_file",
        "description": (
            f"Write text content to a file. Files are saved to {OUTPUT_DIR}/ by default. "
            "Use for saving reports, summaries, or any text output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative paths saved to OUTPUT_DIR)"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "create_excel_report",
        "description": (
            "Create an Excel (.xlsx) report with multiple sheets. "
            "Use to produce structured data reports with tables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sheets_data": {
                    "type": "object",
                    "description": 'Dict of sheet_name -> list of row dicts. E.g. {"Summary": [{"Company": "NVDA", "Score": 9}]}',
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g. report.xlsx)",
                },
            },
            "required": ["sheets_data", "filename"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor factory
# ---------------------------------------------------------------------------

def build_tool_executor():
    """Return an async function that dispatches tool calls by name."""

    async def executor(name: str, inputs: dict) -> Any:
        if name == "web_search":
            return await web_search(inputs["query"], inputs.get("max_results", 10))
        elif name == "fetch_url":
            return await fetch_url(inputs["url"])
        elif name == "read_file":
            return read_file(inputs["path"])
        elif name == "list_files":
            return list_files(inputs.get("directory", "."), inputs.get("pattern", "**/*"))
        elif name == "execute_python":
            return execute_python(inputs["code"])
        elif name == "write_file":
            return write_file(inputs["path"], inputs["content"])
        elif name == "create_excel_report":
            return create_excel_report(inputs["sheets_data"], inputs["filename"])
        else:
            return f"Unknown tool: {name}"

    return executor
