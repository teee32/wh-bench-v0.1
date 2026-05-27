"""Pydantic schemas for QA items and source documents."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    """A single source regulation document."""

    doc_id: str = Field(..., description="Stable short id, e.g. mem-001")
    title: str
    authority: Literal[
        "应急管理部",
        "国家市场监督管理总局",
        "国务院安委办",
        "工信部",
        "其他",
    ]
    url: str
    fetched_at: str = ""  # ISO date
    local_path: str = ""  # relative to data/raw
    content_type: Literal["html", "pdf", "txt"] = "html"
    notes: str = ""


class QAItem(BaseModel):
    """One QA pair in the benchmark."""

    id: str = Field(..., description="e.g. wh-bench-0001")
    question: str
    answer: str
    source_doc: str  # title
    source_url: str
    source_section: str = ""
    source_text: str  # the original snippet the QA was derived from
    category: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    review_status: Literal[
        "auto",            # 仅自动过滤通过
        "human_verified",  # 人工抽检通过
        "human_rejected",  # 人工抽检否决（不入正式集，留 audit）
    ] = "auto"
    review_note: str = ""
