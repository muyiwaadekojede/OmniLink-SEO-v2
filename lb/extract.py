"""Turn HTML / Markdown / MDX / JSX / TSX / Astro into clean text blocks with source mapping.

Features:
- Full HTML DOM parsing with Reasonable Surfer link metadata (anchor text, nav vs. body position, rel attributes)
- JSON-LD Schema.org structured entity extraction (@type, name, headline, author)
- Markdown & MDX parsing with YAML frontmatter extraction and relative + absolute link capture
- Next.js / React (.tsx, .jsx) and Astro (.astro) component static prose & metadata extraction
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

BLOCK_TAGS = {
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "td", "th", "figcaption", "dd", "dt", "summary",
}
DROP_TAGS = {
    "script", "style", "noscript", "svg", "iframe", "nav", "footer",
    "header", "aside", "form", "button", "select", "template",
}
NAV_CONTAINER_TAGS = {"nav", "header", "footer", "aside"}

_WS = re.compile(r"\s+")


@dataclass
class Block:
    text: str
    kind: str
    index: int
    raw: str | None = None
    line: int | None = None

    def as_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "index": self.index, "line": self.line}


@dataclass
class LinkDetail:
    """Rich metadata for an extracted hyperlink (supports Reasonable Surfer & Anchor Entropy)."""
    href: str
    anchor: str = ""
    is_nav: bool = False
    is_external: bool = False
    rel: str = ""
    position_ratio: float = 0.5  # 0.0 = top of document, 1.0 = bottom

    def as_dict(self) -> dict:
        return {
            "href": self.href,
            "anchor": self.anchor,
            "is_nav": self.is_nav,
            "is_external": self.is_external,
            "rel": self.rel,
            "position_ratio": round(self.position_ratio, 3),
        }


@dataclass
class Doc:
    title: str
    description: str
    blocks: list[Block]
    links: list[str]
    h1: str = ""
    link_details: list[LinkDetail] = field(default_factory=list)
    schema_entities: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def words(self) -> int:
        return len(self.text.split())

    def headings(self) -> list[str]:
        return [b.text for b in self.blocks if b.kind.startswith("h")]


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _extract_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Extracts Schema.org JSON-LD entities before script tags are decomposed."""
    entities: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        raw_json = script.string or script.get_text()
        if not raw_json or not raw_json.strip():
            continue
        try:
            data = json.loads(raw_json.strip())
            items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                if "@graph" in item and isinstance(item["@graph"], list):
                    for g_item in item["@graph"]:
                        if isinstance(g_item, dict):
                            entities.append({
                                "type": g_item.get("@type", ""),
                                "name": g_item.get("name") or g_item.get("headline", ""),
                                "url": g_item.get("url", ""),
                            })
                else:
                    entities.append({
                        "type": item.get("@type", ""),
                        "name": item.get("name") or item.get("headline", ""),
                        "url": item.get("url", ""),
                    })
        except Exception:
            continue
    return entities


def parse_html(html: str) -> Doc:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = _clean(soup.title.string)

    description = ""
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if meta and meta.get("content"):
        description = _clean(meta["content"])
    if not description:
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            description = _clean(og["content"])

    schema_entities = _extract_json_ld(soup)

    links: list[str] = []
    link_details: list[LinkDetail] = []
    all_a = soup.find_all("a", href=True)
    total_a = max(1, len(all_a))

    for idx_a, a in enumerate(all_a):
        href = str(a["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        links.append(href)
        anchor_txt = _clean(a.get_text(" ", strip=True))
        is_nav = any(
            (p.name in NAV_CONTAINER_TAGS or p.get("role") == "navigation")
            for p in a.parents
            if getattr(p, "name", None)
        )
        raw_rel = a.get("rel")
        rel_str = " ".join(raw_rel) if isinstance(raw_rel, list) else str(raw_rel or "")
        is_ext = href.startswith(("http://", "https://"))
        link_details.append(
            LinkDetail(
                href=href,
                anchor=anchor_txt,
                is_nav=is_nav,
                is_external=is_ext,
                rel=rel_str.strip(),
                position_ratio=idx_a / total_a,
            )
        )

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()

    blocks: list[Block] = []
    seen: set[str] = set()
    for el in soup.find_all(BLOCK_TAGS):
        # skip blocks nested inside another block we already captured
        if any(p.name in BLOCK_TAGS for p in el.parents):
            continue
        text = _clean(el.get_text(" ", strip=True))
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        if len(text) < 12:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        blocks.append(Block(text=text, kind=el.name, index=len(blocks)))

    h1 = next((b.text for b in blocks if b.kind == "h1"), "")
    return Doc(
        title=title,
        description=description,
        blocks=blocks,
        links=links,
        h1=h1,
        link_details=link_details,
        schema_entities=schema_entities,
    )


def parse_markdown(md: str) -> Doc:
    lines = md.splitlines()

    fm_title = ""
    fm_desc = ""
    fm_h1 = ""

    # Parse and strip YAML frontmatter
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), 80)):
            if lines[i].strip() == "---":
                for fm_line in lines[1:i]:
                    m_kv = re.match(r"^\s*([a-zA-Z0-9_-]+)\s*:\s*(.+?)\s*$", fm_line)
                    if m_kv:
                        k = m_kv.group(1).lower()
                        v = m_kv.group(2).strip().strip("\"'")
                        if k == "title" and not fm_title:
                            fm_title = v
                        elif k in ("description", "summary", "excerpt") and not fm_desc:
                            fm_desc = v
                        elif k == "h1" and not fm_h1:
                            fm_h1 = v
                start = i + 1
                break

    blocks: list[Block] = []
    links: list[str] = []
    link_details: list[LinkDetail] = []
    in_fence = False
    para: list[str] = []
    para_line = 0
    total_lines = max(1, len(lines) - start)

    def flush() -> None:
        nonlocal para
        if not para:
            return
        text = _clean(" ".join(para))
        para = []
        if text and re.search(r"[A-Za-z0-9]", text) and len(text) >= 12:
            blocks.append(Block(text=text, kind="p", index=len(blocks), line=para_line))

    md_link_re = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
    html_link_re = re.compile(r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I)

    for i in range(start, len(lines)):
        raw = lines[i]
        s = raw.strip()

        if s.startswith("```") or s.startswith("~~~"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not s:
            flush()
            continue
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
            flush()
            continue
        if s.startswith("import ") or s.startswith("export "):
            flush()
            continue

        pos_ratio = (i - start) / total_lines

        # Extract Markdown links [anchor](href) — both relative and absolute
        for m_link in md_link_re.finditer(s):
            anc, href = m_link.group(1).strip(), m_link.group(2).strip()
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                links.append(href)
                link_details.append(
                    LinkDetail(
                        href=href,
                        anchor=_clean(anc),
                        is_nav=False,
                        is_external=href.startswith(("http://", "https://")),
                        position_ratio=pos_ratio,
                    )
                )

        # Extract inline HTML links <a href="...">anchor</a>
        for m_html in html_link_re.finditer(s):
            href, anc = m_html.group(1).strip(), m_html.group(2).strip()
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                links.append(href)
                link_details.append(
                    LinkDetail(
                        href=href,
                        anchor=_clean(re.sub(r"<[^>]+>", "", anc)),
                        is_nav=False,
                        is_external=href.startswith(("http://", "https://")),
                        position_ratio=pos_ratio,
                    )
                )

        if s.startswith("<") and not html_link_re.search(s):
            flush()
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            flush()
            heading_txt = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", m.group(2))
            blocks.append(
                Block(text=_clean(heading_txt), kind=f"h{len(m.group(1))}",
                      index=len(blocks), line=i + 1)
            )
            continue
        if re.match(r"^([-*+]|\d+[.)])\s+", s):
            flush()
            text = re.sub(r"^([-*+]|\d+[.)])\s+", "", s)
            text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
            text = re.sub(r"[*_`]{1,3}(.+?)[*_`]{1,3}", r"\1", text)
            text = _clean(text)
            if text and len(text) >= 12:
                blocks.append(Block(text=text, kind="li", index=len(blocks), line=i + 1))
            continue

        if not para:
            para_line = i + 1
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[*_`]{1,3}(.+?)[*_`]{1,3}", r"\1", text)
        para.append(text)
    flush()

    h1 = fm_h1 or next((b.text for b in blocks if b.kind == "h1"), "")
    title = fm_title or h1
    return Doc(
        title=title,
        description=fm_desc,
        blocks=blocks,
        links=links,
        h1=h1,
        link_details=link_details,
    )


def parse_jsx(source: str) -> Doc:
    """Extracts metadata, links, and prose blocks from Next.js / React (.tsx/.jsx) and Astro (.astro) files."""
    title = ""
    description = ""

    # 1. Match Next.js metadata object: title: "..." / description: "..."
    m_title = re.search(r"\btitle\s*:\s*[\"'\`]([^\"'\`\n]+)[\"'\`]", source)
    if m_title:
        title = _clean(m_title.group(1))
    m_desc = re.search(r"\bdescription\s*:\s*[\"'\`]([^\"'\`\n]+)[\"'\`]", source)
    if m_desc:
        description = _clean(m_desc.group(1))

    # 2. Extract <Link href="...">...</Link> and <a href="...">...</a>
    links: list[str] = []
    link_details: list[LinkDetail] = []
    link_re = re.compile(
        r"<(?:Link|a)\b[^>]*\bhref=(?:[\"']([^\"']+)[\"']|\{[\"']([^\"']+)[\"']\})[^>]*>(.*?)</(?:Link|a)>",
        re.S | re.I,
    )
    matches = list(link_re.finditer(source))
    total_m = max(1, len(matches))
    for idx_m, m in enumerate(matches):
        href = (m.group(1) or m.group(2) or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        inner = re.sub(r"<[^>]+>", " ", m.group(3))
        inner = re.sub(r"\{[^}]*\}", " ", inner)
        anc = _clean(inner)
        links.append(href)
        link_details.append(
            LinkDetail(
                href=href,
                anchor=anc,
                is_nav=False,
                is_external=href.startswith(("http://", "https://")),
                position_ratio=idx_m / total_m,
            )
        )

    # 3. Extract JSX prose tags (<h1..h6>, <p>, <li>, <blockquote>)
    blocks: list[Block] = []
    seen: set[str] = set()
    tag_re = re.compile(r"<(h[1-6]|p|li|blockquote)\b[^>]*>(.*?)</\1>", re.S | re.I)
    for m in tag_re.finditer(source):
        kind = m.group(1).lower()
        inner = m.group(2)
        # Strip nested HTML/JSX tags and simple JS expressions
        inner = re.sub(r"<[^>]+>", " ", inner)
        inner = re.sub(r"\{[^{}]*\}", " ", inner)
        text = _clean(inner)
        if not text or len(text) < 12 or not re.search(r"[A-Za-z0-9]", text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        line_num = source[: m.start()].count("\n") + 1
        blocks.append(Block(text=text, kind=kind, index=len(blocks), line=line_num))

    h1 = next((b.text for b in blocks if b.kind == "h1"), "")
    if not title:
        title = h1
    return Doc(
        title=title,
        description=description,
        blocks=blocks,
        links=links,
        h1=h1,
        link_details=link_details,
    )


def parse_file(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    if suffix in {".md", ".mdx", ".markdown"}:
        return parse_markdown(raw)
    if suffix in {".tsx", ".jsx", ".astro"}:
        return parse_jsx(raw)
    return parse_html(raw)


def chunk_blocks(blocks: list[Block], target_words: int = 240) -> list[list[Block]]:
    """Group consecutive blocks so the model sees ~one section of context."""
    chunks: list[list[Block]] = []
    cur: list[Block] = []
    words = 0
    for b in blocks:
        w = len(b.text.split())
        if cur and words + w > target_words:
            chunks.append(cur)
            cur, words = [], 0
        cur.append(b)
        words += w
    if cur:
        chunks.append(cur)
    return chunks
