"""Source corpora to draw from.

v0.1 strategy: rather than crawling government sites (which often 405-block),
we draw from already-published, license-clean Chinese law/regulation corpora
on HuggingFace, then filter by topic keywords.

This is faster, more reproducible, and avoids legal grey zones.
"""
from __future__ import annotations

# ── HuggingFace source corpora ──────────────────────────────────────────
HF_SOURCES: list[dict] = [
    {
        "name": "twang2218",
        "hf_repo": "twang2218/chinese-law-and-regulations",
        "config": "default",
        "split": "train",
        "license": "apache-2.0",
        "fields": {
            "title": "title",
            "content": "content",
            "office": "office",
            "office_category": "office_category",
            "type": "type",
            "publish_date": "publish_date",
        },
        "notes": "22552 PRC laws/regulations from NPC + State Council + ministries",
    },
]


# ── Topic filters ───────────────────────────────────────────────────────
# 标题命中其一才采集（白名单）
TITLE_INCLUDE_KEYWORDS: list[str] = [
    # 安全规程/规范
    "安全生产", "安全规程", "安全规范", "安全技术", "安全管理",
    "操作规程", "作业规程", "技术规范", "技术规程",
    # 高危场景
    "受限空间", "动火", "高处", "粉尘", "有毒",
    "易燃", "易爆", "危险化学品", "危险作业",
    "电气安全", "机械安全", "锅炉", "压力容器",
    "起重", "矿山", "建筑", "施工",
    # 应急 / 防护
    "应急", "防护", "消防", "职业病", "职业健康",
]

# 标题命中任一即排除（黑名单）
TITLE_EXCLUDE_KEYWORDS: list[str] = [
    "公告", "通报", "通知", "答复", "新闻发布",
    "公示", "情况说明", "致辞", "讲话",
    "人事任免", "废止", "决定废止",
]

# office_category 偏好（命中加权）
PREFERRED_OFFICE_CATEGORIES: list[str] = [
    "国务院组成部门",
    "国务院直属机构",
    "国务院特设机构",
    "国务院",
]

# v0.1 目标采集量
V01_TARGET_DOCS = 50  # 过滤完留 30-50 篇做蒸馏
V01_MIN_CONTENT_LEN = 1000   # 太短的文档（公告、目录）跳过
V01_MAX_CONTENT_LEN = 80_000  # 太长的截断，避免单篇蒸馏失控
