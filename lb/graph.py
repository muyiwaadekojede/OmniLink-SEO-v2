"""Topological graph analysis, Reasonable Surfer PageRank, Kleinberg HITS, Click Depth & Centrality.

Implements peer-reviewed graph algorithms:
1. Markov Chain PageRank with Reasonable Surfer Weighting (Page et al. 1999; Dean et al. US Patent 7,716,225 B1)
2. Kleinberg HITS Hubs & Authorities (Kleinberg 1999, JACM)
3. Brandes' Shortest-Path Betweenness Centrality for Structural Bridge Identification (Brandes 2001)
4. Breadth-First Search (BFS) Click Depth & Crawl Reachability Simulation
5. Dead-End Sink Detection & Existing Anchor Text Distribution Histograms
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from .site import Page, SiteIndex


@dataclass
class GraphMetrics:
    page_rank: Dict[str, float]
    projected_page_rank: Dict[str, float]
    in_degrees: Dict[str, int]
    out_degrees: Dict[str, int]
    orphan_pages: List[str]
    under_linked_pages: List[str]
    clusters: Dict[str, List[str]]
    anchor_distributions: Dict[str, Dict[str, int]]
    click_depth: Dict[str, int] = field(default_factory=dict)
    projected_click_depth: Dict[str, int] = field(default_factory=dict)
    hub_scores: Dict[str, float] = field(default_factory=dict)
    authority_scores: Dict[str, float] = field(default_factory=dict)
    betweenness_centrality: Dict[str, float] = field(default_factory=dict)
    dead_end_pages: List[str] = field(default_factory=list)


class LinkGraph:
    """Simulates internal PageRank, Reasonable Surfer equity, HITS, and click depth across the site."""

    def __init__(
        self,
        idx: SiteIndex,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        reasonable_surfer: bool = True,
    ) -> None:
        self.idx = idx
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.reasonable_surfer = reasonable_surfer
        self.pages = idx.pages
        self.n = len(self.pages)
        self.key_to_idx = {p.key: i for i, p in enumerate(self.pages)}

    def build_adjacency(self, additional_links: List[Tuple[str, str]] | None = None) -> np.ndarray:
        """Builds directed adjacency matrix A where A[i, j] > 0 if page i links to page j.

        Under the Reasonable Surfer model (US Patent 7,716,225 B1):
        - Contextual in-body links receive weight in [0.75, 1.0] (higher near top of article).
        - Boilerplate nav/header/footer links receive weight 0.25.
        """
        adj = np.zeros((self.n, self.n), dtype=np.float32)

        from .match import resolve

        for i, page in enumerate(self.pages):
            if self.reasonable_surfer and getattr(page, "link_details", None):
                for ld in page.link_details:
                    target = resolve(ld.href, page, self.idx)
                    if target and target.key in self.key_to_idx and target.key != page.key:
                        j = self.key_to_idx[target.key]
                        if ld.is_nav:
                            w = 0.25
                        else:
                            pos = max(0.0, min(1.0, getattr(ld, "position_ratio", 0.5)))
                            w = 1.0 - 0.25 * pos
                        if w > adj[i, j]:
                            adj[i, j] = w
            else:
                for href in page.links:
                    target = resolve(href, page, self.idx)
                    if target and target.key in self.key_to_idx and target.key != page.key:
                        j = self.key_to_idx[target.key]
                        adj[i, j] = max(adj[i, j], 1.0)

        if additional_links:
            for src_key, tgt_key in additional_links:
                if src_key in self.key_to_idx and tgt_key in self.key_to_idx and src_key != tgt_key:
                    i = self.key_to_idx[src_key]
                    j = self.key_to_idx[tgt_key]
                    adj[i, j] = max(adj[i, j], 1.0)

        return adj

    def compute_pagerank(
        self,
        adj: np.ndarray,
        personalization: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes PageRank (or Topic-Sensitive PageRank when personalization is supplied)."""
        if self.n == 0:
            return np.array([])
        if self.n == 1:
            return np.array([1.0])

        out_weights = adj.sum(axis=1)
        M = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            if out_weights[i] > 0:
                M[:, i] = adj[i, :] / out_weights[i]
            else:
                M[:, i] = 1.0 / self.n

        pr = np.full(self.n, 1.0 / self.n, dtype=np.float32)
        if personalization is not None and personalization.sum() > 0:
            v = (personalization / personalization.sum()).astype(np.float32)
            teleport = (1.0 - self.damping) * v
        else:
            teleport = np.full(self.n, (1.0 - self.damping) / self.n, dtype=np.float32)

        for _ in range(self.max_iter):
            new_pr = self.damping * (M @ pr) + teleport
            if np.sum(np.abs(new_pr - pr)) < self.tol:
                pr = new_pr
                break
            pr = new_pr

        if pr.sum() > 0:
            pr = (pr / pr.sum()) * self.n
        return pr

    def compute_hits(self, adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Kleinberg HITS Hub and Authority scores via power iteration (Kleinberg 1999)."""
        if self.n == 0:
            return np.array([]), np.array([])
        if self.n == 1:
            return np.array([1.0]), np.array([1.0])

        bin_adj = (adj > 0).astype(np.float32)
        hubs = np.ones(self.n, dtype=np.float32)
        auths = np.ones(self.n, dtype=np.float32)

        for _ in range(self.max_iter):
            new_auths = bin_adj.T @ hubs
            norm_a = np.linalg.norm(new_auths)
            if norm_a > 0:
                new_auths /= norm_a

            new_hubs = bin_adj @ new_auths
            norm_h = np.linalg.norm(new_hubs)
            if norm_h > 0:
                new_hubs /= norm_h

            if np.sum(np.abs(new_auths - auths)) + np.sum(np.abs(new_hubs - hubs)) < self.tol:
                auths, hubs = new_auths, new_hubs
                break
            auths, hubs = new_auths, new_hubs

        if hubs.max() > 0:
            hubs = hubs / hubs.max()
        if auths.max() > 0:
            auths = auths / auths.max()
        return hubs, auths

    def compute_click_depth(self, adj: np.ndarray) -> Dict[str, int]:
        """Computes shortest-path click depth from Homepage '/' via Breadth-First Search."""
        if self.n == 0:
            return {}

        root_idx = self.key_to_idx.get("/")
        if root_idx is None:
            # Fallback to page with shortest key or highest out-degree
            root_idx = min(range(self.n), key=lambda idx: len(self.pages[idx].key))

        depths = {p.key: 999 for p in self.pages}
        root_key = self.pages[root_idx].key
        depths[root_key] = 0

        queue = deque([root_idx])
        visited = {root_idx}

        while queue:
            u = queue.popleft()
            u_depth = depths[self.pages[u].key]
            for v in range(self.n):
                if adj[u, v] > 0 and v not in visited:
                    visited.add(v)
                    depths[self.pages[v].key] = u_depth + 1
                    queue.append(v)

        return depths

    def compute_betweenness(self, adj: np.ndarray) -> Dict[str, float]:
        """Computes Brandes' shortest-path betweenness centrality (Brandes 2001)."""
        cb = np.zeros(self.n, dtype=np.float32)
        if self.n <= 2:
            return {p.key: 0.0 for p in self.pages}

        neighbors = [np.where(adj[i, :] > 0)[0].tolist() for i in range(self.n)]

        for s in range(self.n):
            stack: List[int] = []
            pred: List[List[int]] = [[] for _ in range(self.n)]
            sigma = np.zeros(self.n, dtype=np.float32)
            sigma[s] = 1.0
            dist = np.full(self.n, -1, dtype=np.int32)
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in neighbors[v]:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = np.zeros(self.n, dtype=np.float32)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    cb[w] += delta[w]

        norm = float((self.n - 1) * (self.n - 2))
        if norm > 0:
            cb /= norm

        return {self.pages[i].key: round(float(cb[i]), 4) for i in range(self.n)}

    def detect_clusters(self) -> Dict[str, List[str]]:
        """Groups pages into semantic and path-based topic clusters (silos)."""
        clusters: Dict[str, List[str]] = {}
        for p in self.pages:
            parts = [part for part in p.key.strip("/").split("/") if part]
            if len(parts) >= 1:
                cluster_name = f"/{parts[0]}/"
            else:
                cluster_name = "/"
            clusters.setdefault(cluster_name, []).append(p.key)
        return clusters

    def collect_existing_anchors(self) -> Dict[str, Dict[str, int]]:
        """Extracts existing inbound anchor text frequency distributions across the site."""
        from .match import resolve

        anchor_dist: Dict[str, Dict[str, int]] = {p.key: {} for p in self.pages}
        for page in self.pages:
            for ld in getattr(page, "link_details", []):
                if ld.is_nav or not ld.anchor:
                    continue
                target = resolve(ld.href, page, self.idx)
                if target and target.key in anchor_dist and target.key != page.key:
                    norm_anc = ld.anchor.strip().lower()
                    if norm_anc:
                        anchor_dist[target.key][norm_anc] = anchor_dist[target.key].get(norm_anc, 0) + 1
        return anchor_dist

    def analyze(self, additional_links: List[Tuple[str, str]] | None = None) -> GraphMetrics:
        """Runs complete graph analysis including baseline and projected PageRank, HITS, and Click Depth."""
        base_adj = self.build_adjacency()
        base_pr = self.compute_pagerank(base_adj)

        proj_adj = self.build_adjacency(additional_links) if additional_links else base_adj
        proj_pr = self.compute_pagerank(proj_adj)

        bin_base = (base_adj > 0).astype(np.int32)
        in_degrees = {p.key: int(bin_base[:, i].sum()) for i, p in enumerate(self.pages)}
        out_degrees = {p.key: int(bin_base[i, :].sum()) for i, p in enumerate(self.pages)}

        orphans = [p.key for p in self.pages if in_degrees[p.key] == 0 and p.key != "/"]
        under_linked = [p.key for p in self.pages if in_degrees[p.key] == 1 and p.key != "/"]
        dead_ends = [p.key for p in self.pages if out_degrees[p.key] == 0]

        # Sort by lowest in-degree then lowest PageRank
        orphans.sort(key=lambda k: base_pr[self.key_to_idx[k]])
        under_linked.sort(key=lambda k: base_pr[self.key_to_idx[k]])

        pr_dict = {p.key: round(float(base_pr[i]), 4) for i, p in enumerate(self.pages)}
        proj_pr_dict = {p.key: round(float(proj_pr[i]), 4) for i, p in enumerate(self.pages)}

        hubs, auths = self.compute_hits(base_adj)
        hub_dict = {p.key: round(float(hubs[i]), 4) for i, p in enumerate(self.pages)}
        auth_dict = {p.key: round(float(auths[i]), 4) for i, p in enumerate(self.pages)}

        click_depth = self.compute_click_depth(base_adj)
        proj_click_depth = self.compute_click_depth(proj_adj) if additional_links else click_depth
        betweenness = self.compute_betweenness(base_adj)
        anchor_dist = self.collect_existing_anchors()

        return GraphMetrics(
            page_rank=pr_dict,
            projected_page_rank=proj_pr_dict,
            in_degrees=in_degrees,
            out_degrees=out_degrees,
            orphan_pages=orphans,
            under_linked_pages=under_linked,
            clusters=self.detect_clusters(),
            anchor_distributions=anchor_dist,
            click_depth=click_depth,
            projected_click_depth=proj_click_depth,
            hub_scores=hub_dict,
            authority_scores=auth_dict,
            betweenness_centrality=betweenness,
            dead_end_pages=dead_ends,
        )
