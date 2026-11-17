"""Topic cluster detection, Topic-Sensitive PageRank (Haveliwala 2002), silo enforcement, and anchor entropy.

Implements:
- Louvain, Label Propagation, URL Hierarchy, and Hybrid Semantic Community Detection
- Topic-Sensitive PageRank (Haveliwala, WWW 2002) with per-cluster teleportation bias vectors
- Shannon Entropy (Shannon 1948) calculation across combined existing + proposed anchor distributions
- Automated Anchor Variation Generator for Penguin over-optimization remediation
- Keyword Cannibalization & Striking Distance Intelligence from Search Console telemetry
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .site import Page, SiteIndex
from .graph import LinkGraph, GraphMetrics
from .match import resolve


# ---------------------------------------------------------------------------
# Anchor Diversity & Entropy
# ---------------------------------------------------------------------------

@dataclass
class AnchorProfile:
    """Anchor text distribution analysis for a single target page."""
    target_key: str
    target_title: str
    anchors: Dict[str, int]           # anchor_text -> count
    entropy: float                     # Shannon entropy (bits)
    exact_match_ratio: float          # fraction of anchors that are exact title match
    is_over_optimized: bool           # True if entropy < threshold or exact_match > 0.60
    recommendation: str               # human-readable guidance
    suggested_variations: List[str] = field(default_factory=list)


def compute_anchor_entropy(anchor_counts: Dict[str, int]) -> float:
    """Shannon entropy in bits across anchor text variations for a target page."""
    total = sum(anchor_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in anchor_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _generate_anchor_variations(page: Page, used_anchors: Set[str]) -> List[str]:
    """Generates natural partial-match, topical, and contextual anchor variations for a target page."""
    candidates: List[str] = []
    clean_title = re.split(r"\s+[|\-–—:]\s+", page.title or "")[0].strip()
    if clean_title:
        candidates.append(f"{clean_title.lower()} guide")
        candidates.append(f"complete {clean_title.lower()} overview")

    if page.h1:
        clean_h1 = re.split(r"\s+[|\-–—:]\s+", page.h1)[0].strip().lower()
        if clean_h1 and clean_h1 not in used_anchors:
            candidates.append(clean_h1)

    for h in page.headings[:6]:
        clean_h = re.split(r"\s+[|\-–—:]\s+", h)[0].strip().lower()
        if 2 <= len(clean_h.split()) <= 5 and clean_h not in used_anchors:
            candidates.append(clean_h)

    slug_words = page.key.strip("/").split("/")[-1].replace("-", " ").replace("_", " ").strip().lower()
    if slug_words and len(slug_words.split()) >= 2:
        if slug_words not in used_anchors:
            candidates.append(slug_words)
        candidates.append(f"how {slug_words} works")

    unique: List[str] = []
    seen = set(used_anchors)
    for c in candidates:
        if c and c not in seen and len(c) >= 4:
            seen.add(c)
            unique.append(c)
    return unique[:5]


def analyze_anchor_diversity(
    idx: SiteIndex,
    proposed_links: List[Tuple[str, str, str]],  # [(source_key, target_key, anchor_text)]
    min_entropy: float = 1.5,
    max_exact_match_ratio: float = 0.60,
    include_existing: bool = True,
) -> List[AnchorProfile]:
    """Analyze anchor text diversity for all target pages.

    Combines existing body link anchors with proposed new anchors.
    Flags targets that are at risk of over-optimization (Google Penguin).
    """
    combined_anchors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # 1. Collect existing body link anchors if present
    if include_existing:
        graph = LinkGraph(idx)
        existing_map = graph.collect_existing_anchors()
        for tgt_key, counts in existing_map.items():
            for anc_txt, cnt in counts.items():
                combined_anchors[tgt_key][anc_txt] += cnt

    # 2. Collect proposed anchors
    for _, target_key, anchor in proposed_links:
        normalized = anchor.strip().lower()
        if normalized:
            combined_anchors[target_key][normalized] += 1

    profiles = []
    for target_key, anchors in combined_anchors.items():
        if not anchors:
            continue
        page = idx.by_key.get(target_key)
        if not page:
            continue

        total = sum(anchors.values())
        entropy = compute_anchor_entropy(anchors)

        # Check exact match ratio against cleaned title, full title, H1, or exact slug phrase
        title_norm = page.title.strip().lower()
        clean_title_norm = re.split(r"\s+[|\-–—:]\s+", title_norm)[0].strip()
        h1_norm = page.h1.strip().lower() if page.h1 else ""
        slug_norm = page.key.strip("/").split("/")[-1].replace("-", " ").strip().lower()

        exact_targets = {t for t in (title_norm, clean_title_norm, h1_norm, slug_norm) if t}
        exact_count = 0
        for anchor_text, count in anchors.items():
            if anchor_text in exact_targets:
                exact_count += count

        exact_ratio = exact_count / total if total > 0 else 0.0
        is_over_optimized = entropy < min_entropy or exact_ratio > max_exact_match_ratio
        variations = _generate_anchor_variations(page, set(anchors.keys()))

        if is_over_optimized:
            var_hint = f" Suggested alternatives: {', '.join(repr(v) for v in variations[:3])}." if variations else ""
            if exact_ratio > max_exact_match_ratio:
                rec = (f"OVER-OPTIMIZED: {exact_ratio:.0%} of anchors match target title/slug. "
                       f"Diversify with partial-match, topical, and branded anchors.{var_hint}")
            else:
                rec = (f"LOW DIVERSITY: Anchor entropy is {entropy:.2f} bits (min {min_entropy}). "
                       f"Add more varied anchor text to avoid algorithmic penalties.{var_hint}")
        else:
            rec = f"HEALTHY: Entropy {entropy:.2f} bits, exact-match ratio {exact_ratio:.0%}."

        profiles.append(AnchorProfile(
            target_key=target_key,
            target_title=page.title,
            anchors=dict(anchors),
            entropy=round(entropy, 3),
            exact_match_ratio=round(exact_ratio, 3),
            is_over_optimized=is_over_optimized,
            recommendation=rec,
            suggested_variations=variations,
        ))

    profiles.sort(key=lambda p: (not p.is_over_optimized, p.entropy))
    return profiles


# ---------------------------------------------------------------------------
# Topic Cluster Detection & Topic-Sensitive PageRank (Haveliwala 2002)
# ---------------------------------------------------------------------------

@dataclass
class TopicCluster:
    """A detected topic cluster (silo) in the site."""
    cluster_id: int
    name: str                      # derived from most common URL prefix or highest-PR page title
    pages: List[str]               # page keys
    hub_page: str                  # highest PageRank page in cluster
    internal_edge_count: int       # links within cluster
    external_edge_count: int       # links leaving cluster
    cohesion_score: float          # internal/(internal+external) ratio
    topical_pagerank: Dict[str, float] = field(default_factory=dict)


def compute_topic_sensitive_pagerank(
    idx: SiteIndex,
    clusters: List[TopicCluster],
    damping: float = 0.85,
) -> Dict[int, Dict[str, float]]:
    """Computes Topic-Sensitive PageRank (Haveliwala, WWW 2002) for each detected topic cluster.

    Biases random surfer teleportation jumps exclusively toward nodes belonging to cluster C_k,
    measuring each page's topical authority relative to that specific domain silo.
    """
    graph = LinkGraph(idx, damping=damping)
    adj = graph.build_adjacency()
    n = len(idx.pages)
    results: Dict[int, Dict[str, float]] = {}

    if n == 0:
        return results

    for cluster in clusters:
        bias = np.zeros(n, dtype=np.float32)
        for pk in cluster.pages:
            if pk in graph.key_to_idx:
                bias[graph.key_to_idx[pk]] = 1.0
        ts_pr = graph.compute_pagerank(adj, personalization=bias)
        cluster_pr = {
            idx.pages[i].key: round(float(ts_pr[i]), 4)
            for i in range(n)
            if idx.pages[i].key in cluster.pages
        }
        cluster.topical_pagerank = cluster_pr
        results[cluster.cluster_id] = {
            idx.pages[i].key: round(float(ts_pr[i]), 4) for i in range(n)
        }

    return results


def detect_topic_clusters(
    idx: SiteIndex,
    method: str = "url_hierarchy",  # "url_hierarchy" | "louvain" | "label_propagation" | "hybrid"
) -> List[TopicCluster]:
    """Detect topical silos using URL hierarchy, graph modularity, or hybrid semantic clustering."""
    graph = LinkGraph(idx)
    adj = graph.build_adjacency()
    metrics = graph.analyze()

    if method == "url_hierarchy":
        clusters = _cluster_by_url_hierarchy(idx, adj, metrics)
    elif method in ("louvain", "label_propagation"):
        clusters = _cluster_by_graph(idx, adj, metrics, method)
    elif method == "hybrid":
        clusters = _cluster_hybrid(idx, adj, metrics)
    else:
        clusters = _cluster_by_url_hierarchy(idx, adj, metrics)

    compute_topic_sensitive_pagerank(idx, clusters)
    return clusters


def _cluster_by_url_hierarchy(
    idx: SiteIndex, adj: np.ndarray, metrics: GraphMetrics
) -> List[TopicCluster]:
    """Group pages by their primary URL path segment."""
    clusters_map: Dict[str, List[int]] = defaultdict(list)

    for i, page in enumerate(idx.pages):
        parts = [p for p in page.key.strip("/").split("/") if p]
        if len(parts) >= 1:
            prefix = f"/{parts[0]}/"
        else:
            prefix = "/"
        clusters_map[prefix].append(i)

    return _build_cluster_objects(idx, adj, metrics, list(clusters_map.values()), list(clusters_map.keys()))


def _cluster_by_graph(
    idx: SiteIndex, adj: np.ndarray, metrics: GraphMetrics, method: str
) -> List[TopicCluster]:
    """Community detection using NetworkX (Louvain or Label Propagation)."""
    try:
        import networkx as nx
    except ImportError:
        return _cluster_by_url_hierarchy(idx, adj, metrics)

    G = nx.DiGraph()
    for page in idx.pages:
        G.add_node(page.key)

    for i in range(len(idx.pages)):
        for j in range(len(idx.pages)):
            if adj[i, j] > 0 and i != j:
                G.add_edge(idx.pages[i].key, idx.pages[j].key, weight=float(adj[i, j]))

    G_undirected = G.to_undirected()

    if method == "louvain":
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G_undirected, seed=42)
        except (ImportError, AttributeError):
            from networkx.algorithms.community import label_propagation_communities
            communities = list(label_propagation_communities(G_undirected))
    else:
        from networkx.algorithms.community import label_propagation_communities
        communities = list(label_propagation_communities(G_undirected))

    key_to_idx = {p.key: i for i, p in enumerate(idx.pages)}
    groups = [[key_to_idx[k] for k in comm if k in key_to_idx] for comm in communities]
    return _build_cluster_objects(idx, adj, metrics, groups)


def _cluster_hybrid(
    idx: SiteIndex, adj: np.ndarray, metrics: GraphMetrics
) -> List[TopicCluster]:
    """Combines hyperlink adjacency, URL path affinity, and semantic vector similarity for Louvain."""
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        return _cluster_by_url_hierarchy(idx, adj, metrics)

    n = len(idx.pages)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    # Compute semantic similarity matrix if vectors available
    sim_matrix = None
    if idx.vectors is not None and len(idx.vectors) == n:
        sim_matrix = idx.vectors @ idx.vectors.T

    for i in range(n):
        parts_i = [p for p in idx.pages[i].key.strip("/").split("/") if p]
        prefix_i = parts_i[0] if parts_i else ""
        for j in range(i + 1, n):
            parts_j = [p for p in idx.pages[j].key.strip("/").split("/") if p]
            prefix_j = parts_j[0] if parts_j else ""

            w = 0.0
            if adj[i, j] > 0 or adj[j, i] > 0:
                w += float(adj[i, j] + adj[j, i]) * 1.5
            if prefix_i and prefix_i == prefix_j:
                w += 1.0
            if sim_matrix is not None and sim_matrix[i, j] > 0.45:
                w += float(sim_matrix[i, j]) * 0.8

            if w > 0.35:
                G.add_edge(i, j, weight=w)

    communities = louvain_communities(G, weight="weight", seed=42)
    groups = [ sorted(list(comm)) for comm in communities if comm ]
    return _build_cluster_objects(idx, adj, metrics, groups)


def _build_cluster_objects(
    idx: SiteIndex,
    adj: np.ndarray,
    metrics: GraphMetrics,
    groups: List[List[int]],
    prefix_hints: Optional[List[str]] = None,
) -> List[TopicCluster]:
    results: List[TopicCluster] = []
    for cid, page_indices in enumerate(groups):
        if not page_indices:
            continue
        page_keys = [idx.pages[i].key for i in page_indices]
        idx_set = set(page_indices)

        internal_edges = 0
        external_edges = 0
        for i in page_indices:
            for j in range(len(idx.pages)):
                if adj[i, j] > 0:
                    if j in idx_set:
                        internal_edges += 1
                    else:
                        external_edges += 1

        total = internal_edges + external_edges
        cohesion = internal_edges / total if total > 0 else 0.0

        hub = max(page_keys, key=lambda k: metrics.page_rank.get(k, 0.0))
        hub_page = idx.by_key.get(hub)
        default_name = prefix_hints[cid] if (prefix_hints and cid < len(prefix_hints)) else f"Cluster {cid}"
        name = hub_page.title[:50] if hub_page and hub_page.title else default_name

        results.append(TopicCluster(
            cluster_id=cid,
            name=name,
            pages=page_keys,
            hub_page=hub,
            internal_edge_count=internal_edges,
            external_edge_count=external_edges,
            cohesion_score=round(cohesion, 3),
        ))

    results.sort(key=lambda c: -len(c.pages))
    for new_id, c in enumerate(results):
        c.cluster_id = new_id
    return results


# ---------------------------------------------------------------------------
# Silo Enforcement Guard
# ---------------------------------------------------------------------------

def check_silo_violations(
    clusters: List[TopicCluster],
    proposed_links: List[Tuple[str, str, str]],  # (source, target, anchor)
) -> List[Dict[str, Any]]:
    """Flag proposed links that cross silo boundaries without hub mediation.

    A cross-silo link is acceptable IF:
    - Source is the hub page of its cluster, OR
    - Target is the hub page of its cluster
    """
    page_to_cluster: Dict[str, int] = {}
    hub_pages: Set[str] = set()

    for cluster in clusters:
        for pk in cluster.pages:
            page_to_cluster[pk] = cluster.cluster_id
        hub_pages.add(cluster.hub_page)

    violations = []
    for source, target, anchor in proposed_links:
        src_cluster = page_to_cluster.get(source)
        tgt_cluster = page_to_cluster.get(target)

        if src_cluster is None or tgt_cluster is None:
            continue

        if src_cluster != tgt_cluster:
            if source not in hub_pages and target not in hub_pages:
                violations.append({
                    "source": source,
                    "target": target,
                    "anchor": anchor,
                    "source_cluster": src_cluster,
                    "target_cluster": tgt_cluster,
                    "severity": "warning",
                    "message": (
                        f"Cross-silo link from cluster {src_cluster} to cluster {tgt_cluster}. "
                        f"Neither endpoint is a hub page. Consider routing through hub pages "
                        f"to preserve topical authority."
                    ),
                })

    return violations


# ---------------------------------------------------------------------------
# Keyword Cannibalization & Striking Distance Intelligence
# ---------------------------------------------------------------------------

@dataclass
class CannibalizationIssue:
    """A keyword cannibalization problem detected from Search Console data."""
    query: str
    competing_pages: List[Dict[str, Any]]  # [{page, clicks, impressions, position}]
    recommendation: str
    severity: str  # "critical" | "warning" | "info"
    primary_page: str = ""
    wasted_impressions: int = 0


def detect_cannibalization(
    search_console_rows: List[Any],
    min_impressions: int = 10,
) -> List[CannibalizationIssue]:
    """Detect keyword cannibalization from Search Console Performance data.

    Cannibalization occurs when 2+ pages compete for the same query,
    splitting clicks and diluting ranking signals.
    """
    from .browser import SearchConsoleRow

    query_pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in search_console_rows:
        if isinstance(row, SearchConsoleRow):
            row = {
                "query": row.query,
                "page": row.page,
                "clicks": row.clicks,
                "impressions": row.impressions,
                "position": row.position,
            }

        if row.get("impressions", 0) < min_impressions:
            continue
        query_pages[row["query"]].append(row)

    issues = []
    for query, pages in query_pages.items():
        if len(pages) < 2:
            continue

        pages_sorted = sorted(pages, key=lambda p: (p.get("position", 999), -p.get("clicks", 0)))
        total_clicks = sum(p.get("clicks", 0) for p in pages_sorted)
        secondary_imps = sum(p.get("impressions", 0) for p in pages_sorted[1:])

        if total_clicks == 0 and secondary_imps < 50:
            severity = "info"
        elif pages_sorted[0].get("position", 999) > 10 or secondary_imps >= 200:
            severity = "critical"
        else:
            severity = "warning"

        best_page = pages_sorted[0]["page"]
        rec = (
            f"Query '{query}' has {len(pages)} competing pages ({secondary_imps} secondary impressions split). "
            f"Consolidate ranking signals to {best_page} (position {pages_sorted[0].get('position', 0.0):.1f}). "
            f"Link from secondary pages TO the primary page using '{query}' as anchor text."
        )

        issues.append(CannibalizationIssue(
            query=query,
            competing_pages=pages_sorted,
            recommendation=rec,
            severity=severity,
            primary_page=best_page,
            wasted_impressions=secondary_imps,
        ))

    issues.sort(key=lambda i: ({"critical": 0, "warning": 1, "info": 2}[i.severity], -i.wasted_impressions))
    return issues
