"""Contract tests for the offline teaching notebook."""

from __future__ import annotations

import json
from pathlib import Path


SECTION_KEYWORDS: tuple[str, ...] = (
    "parquet loading",
    "realized variance",
    "feature building",
    "model training",
    "evaluation",
    "outputs",
    "interpretation",
)


def test_offline_research_notebook_has_markdown_code_pairs() -> None:
    """Notebook should alternate substantial markdown and code cells."""
    notebook_path = Path("notebooks/01_offline_research_pipeline.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    assert cells[0]["cell_type"] == "markdown"

    paired_cells = cells[1:]
    assert len(paired_cells) % 2 == 0

    for index in range(0, len(paired_cells), 2):
        assert paired_cells[index]["cell_type"] == "markdown"
        assert paired_cells[index + 1]["cell_type"] == "code"


def test_offline_research_notebook_sections_and_offline_runtime_statement() -> None:
    """Notebook should cover full pipeline stages and offline runtime note."""
    notebook_path = Path("notebooks/01_offline_research_pipeline.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    markdown_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    ).lower()

    for keyword in SECTION_KEYWORDS:
        assert keyword in markdown_text

    assert "clickhouse is not required" in markdown_text
