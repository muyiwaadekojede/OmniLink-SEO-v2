"""LinkBERT inference: predict natural anchor-text spans in plain text.

The upstream repo ships a Streamlit demo whose offset handling is lossy (it
re-emits text, drops overlaps, and breaks on long documents). This module is a
high-precision, production-grade implementation:

* batched sliding windows with stride, so long pages are covered at high speed
* exact character offsets preserved, so spans map cleanly back to source text
* compound-aware lexical word-boundary snapping (preserving hyphenated & numeric terms)
* bidirectional function-word & unclosed-delimiter trimming
* token-level grouping with hysteresis (high seed / low expand)
* span scoring by mean token probability
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "dejanseo/LinkBERT"

# Heavy artifacts live outside the workspace so OneDrive never syncs them.
CACHE_DIR = os.environ.get(
    "LINKBERT_CACHE",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "linkbert", "models"),
)

_SENT_END = re.compile(r"[.!?\n]")
_EDGE_JUNK = " \t\r\n.,;:!?\"'“”‘’()[]{}<>—–-/\\|*_~`"
_GAP_OK = re.compile(r"^[\s\-–—/&']*$")
_STOP_ANCHORS = {
    "click here", "read more", "learn more", "here", "this", "this article",
    "this page", "this post", "link", "website", "more", "see more", "find out more",
    "get a free call instead", "contact me directly", "requires you to publish the site",
    "page onto google", "source", "reference", "check this out", "visit website",
    "submit your website", "the challenge", "the solution", "the results", "what you get",
}

_CTA_PREFIXES = (
    "get a free", "free call", "contact me", "requires you to",
    "click here", "read more", "page onto", "i implemented", "we implemented",
    "book a call", "schedule a call", "hire me", "let us", "feel free to",
    "reach out", "sign up", "subscribe to", "submit your",
)

_LEADING_DISCARD_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "when", "because", "while",
    "although", "since", "as", "by", "with", "from", "into", "onto", "for",
    "of", "in", "on", "at", "to", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "their", "your", "my", "also", "even",
    "i", "we", "you", "am", "do", "did", "does", "have", "has", "had", "through",
}

_TRAILING_DISCARD_WORDS = {
    "because", "they", "this", "that", "these", "those", "you", "your",
    "and", "or", "but", "is", "was", "are", "were", "be", "been", "have",
    "has", "had", "with", "for", "from", "into", "onto", "hard", "also",
    "the", "a", "an", "of", "in", "on", "at", "to", "by", "as", "if",
    "when", "while", "which", "who", "whom", "whose", "where", "than", "then",
}


@dataclass
class Span:
    text: str
    start: int
    end: int
    score: float
    words: int
    sentence: str
    paragraph: int | None = None

    def as_dict(self) -> dict:
        d = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 4),
            "words": self.words,
            "sentence": self.sentence,
        }
        if self.paragraph is not None:
            d["paragraph"] = self.paragraph
        return d


def _sentence_around(text: str, start: int, end: int, limit: int = 320) -> str:
    left = start
    while left > 0 and start - left < limit:
        if _SENT_END.match(text[left - 1]):
            break
        left -= 1
    right = end
    while right < len(text) and right - end < limit:
        if _SENT_END.match(text[right]):
            right += 1
            break
        right += 1
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _snap_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Snaps character offsets to full whitespace/punctuation and compound word boundaries.

    Prevents BPE/WordPiece subword slicing from cutting off words mid-phrase
    (e.g., 'back' from 'backlinks', 'pen' from 'penalty', or '1,212' / 'accident-only').
    """
    n = len(text)
    while start > 0:
        if text[start - 1].isalnum():
            start -= 1
        elif start > 1 and text[start - 1] in "-,." and text[start - 2].isalnum() and (start < n and text[start].isalnum()):
            start -= 2
        else:
            break

    while end < n:
        if text[end].isalnum():
            end += 1
        elif end + 1 < n and text[end] in "-,." and text[end + 1].isalnum() and (end > 0 and text[end - 1].isalnum()):
            end += 2
        elif text[end] == "%" and end > 0 and text[end - 1].isdigit():
            end += 1
        else:
            break

    return start, end


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    # Snap to lexical word boundaries first
    start, end = _snap_word_boundaries(text, start, end)

    while start < end and text[start] in _EDGE_JUNK:
        start += 1
    while end > start and text[end - 1] in _EDGE_JUNK:
        end -= 1

    # Re-snap in case trimming touched word edge
    start, end = _snap_word_boundaries(text, start, end)

    # Handle unclosed parentheses, brackets, or quotes in the span
    span_txt = text[start:end]
    if "(" in span_txt and ")" not in span_txt:
        idx = span_txt.find("(")
        end = start + idx
    elif ")" in span_txt and "(" not in span_txt:
        idx = span_txt.find(")")
        start = start + idx + 1
    elif "[" in span_txt and "]" not in span_txt:
        idx = span_txt.find("[")
        end = start + idx
    elif '"' in span_txt and span_txt.count('"') % 2 != 0:
        idx = span_txt.rfind('"')
        if idx == 0:
            start += 1
        else:
            end = start + idx

    while start < end and text[start] in _EDGE_JUNK:
        start += 1
    while end > start and text[end - 1] in _EDGE_JUNK:
        end -= 1

    return start, end


class LinkPredictor:
    """High-throughput wrapper over dejanseo/LinkBERT (bert-large-cased token classifier)."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        cache_dir: str = CACHE_DIR,
        max_length: int = 512,
        stride: int = 96,
        batch_size: int = 8,
    ) -> None:
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model, info = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=cache_dir, output_loading_info=True
        )
        missing = [k for k in info.get("missing_keys", []) if k.startswith("classifier")]
        if missing:
            raise RuntimeError(
                "Checkpoint has no token-classification head (missing "
                f"{missing}). The published weights may be masked-LM only."
            )
        self.model = model.to(self.device).eval()

    # ------------------------------------------------------------------ infer
    def _token_probs(self, text: str) -> list[tuple[int, int, float]]:
        """Return (char_start, char_end, p_link) for every real token using batched inference."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            stride=self.stride,
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)

        best: dict[tuple[int, int], float] = {}
        num_windows = offsets.shape[0]
        with torch.no_grad():
            for b_start in range(0, num_windows, self.batch_size):
                b_end = min(num_windows, b_start + self.batch_size)
                batch = {k: v[b_start:b_end].to(self.device) for k, v in enc.items()}
                logits = self.model(**batch).logits.float()
                probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
                for w_idx in range(b_end - b_start):
                    w_offsets = offsets[b_start + w_idx].tolist()
                    w_probs = probs[w_idx].tolist()
                    for (s, e), p in zip(w_offsets, w_probs):
                        if s == e:
                            continue
                        key = (s, e)
                        if p > best.get(key, -1.0):
                            best[key] = p
        return sorted(((s, e, p) for (s, e), p in best.items()), key=lambda t: t[0])

    # ------------------------------------------------------------------ spans
    def predict(
        self,
        text: str,
        high: float = 0.50,
        low: float = 0.22,
        min_words: int = 1,
        max_words: int = 8,
        max_spans: int | None = None,
    ) -> list[Span]:
        if not text or not text.strip():
            return []

        tokens = self._token_probs(text)
        spans: list[Span] = []
        cur: list[tuple[int, int, float]] = []

        def close() -> None:
            nonlocal cur
            if not cur:
                return
            s, e = cur[0][0], cur[-1][1]
            s, e = _trim(text, s, e)
            cur = []
            if s >= e:
                return
            piece = text[s:e]

            # Tokenize with offsets inside `piece` so leading/trailing discard words adjust exact s, e
            word_matches = list(re.finditer(r"\S+", piece))
            while word_matches and word_matches[0].group(0).strip(_EDGE_JUNK).lower() in _LEADING_DISCARD_WORDS:
                word_matches.pop(0)
            while word_matches and word_matches[-1].group(0).strip(_EDGE_JUNK).lower() in _TRAILING_DISCARD_WORDS:
                word_matches.pop()

            if not word_matches or len(word_matches) < min_words:
                return

            new_s = s + word_matches[0].start()
            new_e = s + word_matches[-1].end()
            s, e = _trim(text, new_s, new_e)
            if s >= e:
                return

            piece = text[s:e]
            words_list = piece.split()
            words = len(words_list)
            if words < min_words or words > max_words:
                return
            if not re.search(r"[A-Za-z]", piece):
                return
            clean_lower = piece.strip().lower()
            if clean_lower in _STOP_ANCHORS:
                return
            if any(clean_lower.startswith(pfx) for pfx in _CTA_PREFIXES):
                return
            spans.append(
                Span(
                    text=piece,
                    start=s,
                    end=e,
                    score=0.0,
                    words=words,
                    sentence=_sentence_around(text, s, e),
                )
            )

        for s, e, p in tokens:
            if p >= high:
                if cur and not _GAP_OK.match(text[cur[-1][1] : s]):
                    close()
                cur.append((s, e, p))
            elif p >= low and cur and _GAP_OK.match(text[cur[-1][1] : s]):
                cur.append((s, e, p))
            else:
                close()
        close()

        # score spans properly now that boundaries are final
        for sp in spans:
            covered = [p for s, e, p in tokens if s >= sp.start and e <= sp.end]
            sp.score = float(np.mean(covered)) if covered else 0.0

        # merge spans that touch across a whitespace/hyphen gap
        merged: list[Span] = []
        for sp in spans:
            if merged and _GAP_OK.match(text[merged[-1].end : sp.start]) and merged[-1].end <= sp.start:
                prev = merged[-1]
                new_text = text[prev.start : sp.end]
                if len(new_text.split()) <= max_words:
                    prev.text = new_text
                    prev.end = sp.end
                    prev.words = len(new_text.split())
                    prev.score = max(prev.score, sp.score)
                    prev.sentence = _sentence_around(text, prev.start, prev.end)
                    continue
            merged.append(sp)

        merged.sort(key=lambda s: (-s.score, s.start))
        if max_spans:
            merged = merged[:max_spans]
        merged.sort(key=lambda s: s.start)
        return merged

    def annotate(self, text: str, **kwargs) -> str:
        """Underline predicted spans (same visual contract as the upstream demo)."""
        spans = self.predict(text, **kwargs)
        out, prev = [], 0
        for sp in spans:
            if sp.start < prev:
                continue
            out.append(text[prev : sp.start])
            out.append(f"<u>{text[sp.start:sp.end]}</u>")
            prev = sp.end
        out.append(text[prev:])
        return "".join(out)
