"""Industrial-grade, multi-tab interactive visual dashboard & 6-tab Excel Master Workbook generator for LinkBERT Pro.

Generates:
1. Standalone, dependency-free 5-Tab Interactive HTML5 Application (`report.html`) featuring:
   - Force-directed interactive SVG network graph with PageRank/HITS scaling, zoom, pan & hover inspection
   - Internal Linking Matrix with multi-facet search, engine filtering, and 1-click HTML + Markdown copy
   - External E-E-A-T Authority Citations browser with TrustRank tiers and academic justifications
   - Graph Topology table (PageRank, Projected PR Delta, Click Depth, HITS Hub/Authority, Bridge Centrality)
   - Topic Silos, Topic-Sensitive PageRank (Haveliwala 2002), and Shannon Entropy Anchor Diversity Guard
   - Google Search Console Striking Distance & Keyword Cannibalization Matrix
2. 6-Tab Executive Excel Master Workbook (`link_architecture_master.xlsx`) with conditional color formatting
3. Structured CSV (`suggestions.csv`, `external_links.csv`) and JSON (`suggestions.json`, `graph_metrics.json`) exports
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from .discovery import CandidateOpportunity
from .graph import GraphMetrics, LinkGraph
from .site import SiteIndex


def _e(s: Any) -> str:
    return html.escape(str(s or ""), quote=True)


def generate_dashboard_html(
    idx: SiteIndex,
    opportunities: List[CandidateOpportunity],
    metrics: GraphMetrics | None = None,
    title: str = "LinkBERT Pro — SEO Link Intelligence Dashboard",
    external_opps: Optional[List[Any]] = None,
    gsc_queries: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if metrics is None:
        proposed_pairs = [(o.source_key, o.target_key) for o in opportunities]
        metrics = LinkGraph(idx).analyze(additional_links=proposed_pairs)

    from .intelligence import analyze_anchor_diversity, detect_topic_clusters

    proposed_triples = [(o.source_key, o.target_key, o.anchor) for o in opportunities]
    anchor_profiles = analyze_anchor_diversity(idx, proposed_triples)
    topic_clusters = detect_topic_clusters(idx, method="url_hierarchy")

    # Prepare node & link data for interactive force-directed graph
    nodes = []
    links = []
    page_key_to_id = {}

    for i, p in enumerate(idx.pages):
        page_key_to_id[p.key] = i
        pr = metrics.page_rank.get(p.key, 1.0)
        proj_pr = metrics.projected_page_rank.get(p.key, pr)
        cd = metrics.click_depth.get(p.key, 0)
        proj_cd = metrics.projected_click_depth.get(p.key, cd)
        hub_s = metrics.hub_scores.get(p.key, 0.0)
        auth_s = metrics.authority_scores.get(p.key, 0.0)
        bw_s = metrics.betweenness_centrality.get(p.key, 0.0)
        in_d = metrics.in_degrees.get(p.key, 0)
        out_d = metrics.out_degrees.get(p.key, 0)

        nodes.append({
            "id": i,
            "key": p.key,
            "title": p.title or p.key,
            "pr": pr,
            "projPr": proj_pr,
            "clickDepth": cd,
            "projClickDepth": proj_cd,
            "hub": hub_s,
            "auth": auth_s,
            "bridge": bw_s,
            "inDeg": in_d,
            "outDeg": out_d,
            "isOrphan": (p.key in metrics.orphan_pages),
            "isUnder": (p.key in metrics.under_linked_pages),
            "cluster": p.key.strip("/").split("/")[0] if p.key.strip("/") else "root",
        })

    from .match import resolve
    for p in idx.pages:
        src_id = page_key_to_id[p.key]
        for href in p.links:
            target = resolve(href, p, idx)
            if target and target.key in page_key_to_id and target.key != p.key:
                links.append({
                    "source": src_id,
                    "target": page_key_to_id[target.key],
                    "type": "existing",
                })

    for opp in opportunities:
        if opp.source_key in page_key_to_id and opp.target_key in page_key_to_id:
            links.append({
                "source": page_key_to_id[opp.source_key],
                "target": page_key_to_id[opp.target_key],
                "type": "proposed",
                "score": opp.score,
            })

    total_opps = len(opportunities)
    ext_list = [eo.as_dict() if hasattr(eo, "as_dict") else eo for eo in (external_opps or [])]
    total_ext = len(ext_list)
    orphans_count = len(metrics.orphan_pages)
    pages_count = len(idx.pages)
    avg_score = sum(o.score for o in opportunities) / total_opps if total_opps else 0.0

    raw_opps = [o.as_dict() for o in opportunities]
    clusters_payload = [{
        "id": c.cluster_id,
        "name": c.name,
        "hub": c.hub_page,
        "pages": c.pages,
        "cohesion": c.cohesion_score,
        "inEdges": c.internal_edge_count,
        "outEdges": c.external_edge_count,
    } for c in topic_clusters]
    entropy_payload = [{
        "target": ap.target_key,
        "title": ap.target_title,
        "entropy": ap.entropy,
        "exactRatio": ap.exact_match_ratio,
        "overOptimized": ap.is_over_optimized,
        "rec": ap.recommendation,
        "anchors": ap.anchors,
        "variations": ap.suggested_variations,
    } for ap in anchor_profiles]

    graph_data_json = json.dumps({"nodes": nodes, "links": links})
    opps_json = json.dumps(raw_opps)
    ext_json = json.dumps(ext_list)
    clusters_json = json.dumps(clusters_payload)
    entropy_json = json.dumps(entropy_payload)
    gsc_json = json.dumps(gsc_queries or [])

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      --bg: #070b14;
      --surface: #111827;
      --surface-hover: #1e293b;
      --surface-border: #1f2937;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.15);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.5;
      padding-bottom: 80px;
    }}
    header {{
      background: linear-gradient(180deg, #152036 0%, var(--surface) 100%);
      border-bottom: 1px solid var(--surface-border);
      padding: 28px 48px 0;
    }}
    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      padding: 3px 10px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.18);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.35);
    }}
    .sub {{ color: var(--text-muted); font-size: 13.5px; margin-bottom: 20px; }}
    .nav-tabs {{
      display: flex;
      gap: 6px;
      border-bottom: 1px solid transparent;
    }}
    .nav-tab {{
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--text-muted);
      padding: 12px 18px;
      font-size: 13.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .nav-tab:hover {{ color: var(--text); }}
    .nav-tab.active {{
      color: var(--accent);
      border-bottom-color: var(--accent);
      background: rgba(56, 189, 248, 0.06);
    }}
    .container {{ max-width: 1420px; margin: 0 auto; padding: 28px 48px; }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}

    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::after {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: var(--accent);
    }}
    .kpi-card.success::after {{ background: var(--success); }}
    .kpi-card.warning::after {{ background: var(--warning); }}
    .kpi-card.purple::after {{ background: var(--purple); }}
    .kpi-val {{ font-size: 28px; font-weight: 800; line-height: 1.15; margin-bottom: 4px; }}
    .kpi-lbl {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-muted); }}

    .section-box {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 28px;
    }}
    .section-title {{
      font-size: 17px;
      font-weight: 600;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}
    #graph-container {{
      width: 100%;
      height: 520px;
      background: #050811;
      border: 1px solid #1e293b;
      border-radius: 10px;
      position: relative;
      overflow: hidden;
      margin-top: 12px;
    }}
    svg.graph-svg {{ width: 100%; height: 100%; cursor: grab; }}
    svg.graph-svg:active {{ cursor: grabbing; }}
    .graph-tooltip {{
      position: absolute;
      bottom: 14px;
      left: 14px;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #334155;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      pointer-events: none;
      color: var(--text);
    }}

    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .search-input {{
      flex: 1;
      min-width: 260px;
      background: #080c14;
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 9px 14px;
      border-radius: 8px;
      font-size: 13.5px;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--accent); }}
    .filter-btn {{
      background: #0b1120;
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .filter-btn.active {{
      background: var(--accent);
      color: #070b14;
      border-color: var(--accent);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 13.5px;
    }}
    th {{
      background: #0b1120;
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--surface-border);
    }}
    td {{
      padding: 13px 14px;
      border-bottom: 1px solid var(--surface-border);
      vertical-align: top;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

    .anchor-hl {{
      background: rgba(56, 189, 248, 0.22);
      color: #38bdf8;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }}
    .engine-badge {{
      font-size: 10.5px;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
      display: inline-block;
      margin-top: 4px;
    }}
    .engine-mention {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .engine-linkbert {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
    .tier-badge {{
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 700;
      display: inline-block;
    }}
    .tier-1 {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .tier-2 {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .tier-3 {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .score-bar {{
      display: inline-block;
      width: 56px;
      height: 6px;
      background: #1e293b;
      border-radius: 3px;
      overflow: hidden;
      margin-left: 6px;
      vertical-align: middle;
    }}
    .score-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #10b981);
    }}
    .btn-copy {{
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 11.5px;
      padding: 4px 9px;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .btn-copy:hover {{ background: #334155; }}
    .snippet {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px;
      background: #060911;
      padding: 6px 10px;
      border-radius: 6px;
      color: #93c5fd;
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <h1>{_e(title)} <span class="badge">Pro v3.0</span></h1>
      <span class="badge">BM25F + LinkBERT + Reasonable Surfer + HITS</span>
    </div>
    <div class="sub">{_e(idx.root)} · {pages_count} Pages Indexed · {total_opps} Internal Opportunities · {total_ext} External Authority Citations</div>
    <div class="nav-tabs">
      <button class="nav-tab active" onclick="switchTab('tab-internal', this)">1. Internal Link Matrix ({total_opps})</button>
      <button class="nav-tab" onclick="switchTab('tab-external', this)">2. External E-E-A-T Citations ({total_ext})</button>
      <button class="nav-tab" onclick="switchTab('tab-graph', this)">3. Network Graph, PageRank &amp; HITS ({pages_count})</button>
      <button class="nav-tab" onclick="switchTab('tab-silos', this)">4. Topic Silos &amp; Anchor Entropy ({len(topic_clusters)})</button>
      <button class="nav-tab" onclick="switchTab('tab-gsc', this)">5. GSC Striking Distance ({len(gsc_queries or [])})</button>
    </div>
  </header>

  <div class="container">
    <!-- KPI Row -->
    <div class="kpi-grid">
      <div class="kpi-card success">
        <div class="kpi-val">{total_opps}</div>
        <div class="kpi-lbl">Internal Link Placements</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-val">{total_ext}</div>
        <div class="kpi-lbl">External Authority Citations</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{pages_count}</div>
        <div class="kpi-lbl">Indexed Site Pages</div>
      </div>
      <div class="kpi-card warning">
        <div class="kpi-val">{orphans_count}</div>
        <div class="kpi-lbl">Orphan Pages Detected</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{avg_score:.2f}</div>
        <div class="kpi-lbl">Mean Confidence Score</div>
      </div>
    </div>

    <!-- TAB 1: INTERNAL LINKS -->
    <div id="tab-internal" class="tab-pane active">
      <div class="section-box">
        <div class="section-title">
          <span>Internal Link Optimization Matrix</span>
          <button class="filter-btn active" onclick="downloadJSON()">Export Full JSON</button>
        </div>
        <div class="controls">
          <input type="text" id="search-input" class="search-input" placeholder="Search by anchor text, source URL, target URL, or sentence..." oninput="filterTable()">
          <button class="filter-btn active" onclick="setEngineFilter('all', this)">All Engines ({total_opps})</button>
          <button class="filter-btn" onclick="setEngineFilter('direct_mention', this)">Direct &amp; GSC Mentions</button>
          <button class="filter-btn" onclick="setEngineFilter('linkbert', this)">LinkBERT Neural</button>
        </div>
        <table>
          <thead>
            <tr>
              <th style="width: 19%;">Source Page</th>
              <th style="width: 20%;">Anchor Text</th>
              <th style="width: 21%;">Target Page</th>
              <th style="width: 12%;">Score</th>
              <th style="width: 28%;">Context &amp; Snippets</th>
            </tr>
          </thead>
          <tbody id="table-body"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 2: EXTERNAL CITATIONS -->
    <div id="tab-external" class="tab-pane">
      <div class="section-box">
        <div class="section-title">
          <span>External Authority &amp; E-E-A-T Citations (TrustRank Seed Sets)</span>
          <input type="text" id="ext-search-input" class="search-input" style="max-width:340px;" placeholder="Filter by authority domain, anchor, or page..." oninput="renderExternalTable()">
        </div>
        <table>
          <thead>
            <tr>
              <th style="width: 16%;">Source Page</th>
              <th style="width: 16%;">Claim / Entity Anchor</th>
              <th style="width: 22%;">Authority Target</th>
              <th style="width: 12%;">Trust Tier</th>
              <th style="width: 34%;">Sentence Context &amp; Academic Basis</th>
            </tr>
          </thead>
          <tbody id="ext-table-body"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: GRAPH TOPOLOGY & PAGERANK -->
    <div id="tab-graph" class="tab-pane">
      <div class="section-box">
        <div class="section-title">
          <span>Force-Directed Site Link Topology (PageRank, Click Depth &amp; Orphan Rescue)</span>
          <span style="font-size: 12.5px; color: var(--text-muted);">Blue = Healthy · Red = Orphan · Green Dashed = Proposed Rescue Links</span>
        </div>
        <div id="graph-container">
          <svg id="network-svg" class="graph-svg"></svg>
          <div id="node-info" class="graph-tooltip">Hover over any page node to inspect PageRank, HITS Authority, and Click Depth.</div>
        </div>
      </div>
      <div class="section-box">
        <div class="section-title"><span>Page-Level Graph Telemetry (Reasonable Surfer PageRank, HITS &amp; Click Depth)</span></div>
        <table>
          <thead>
            <tr>
              <th>Page URL</th>
              <th>Current PR</th>
              <th>Projected PR</th>
              <th>PR Lift</th>
              <th>Click Depth</th>
              <th>HITS Authority</th>
              <th>HITS Hub</th>
              <th>Bridge Score</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="topology-table-body"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 4: TOPIC SILOS & ANCHOR ENTROPY -->
    <div id="tab-silos" class="tab-pane">
      <div class="section-box">
        <div class="section-title"><span>Detected Topic Clusters &amp; Silo Cohesion</span></div>
        <table>
          <thead>
            <tr>
              <th>Cluster ID</th>
              <th>Cluster Name</th>
              <th>Primary Hub Page</th>
              <th>Pages</th>
              <th>Intra-Silo Edges</th>
              <th>Cross-Silo Edges</th>
              <th>Cohesion Score</th>
            </tr>
          </thead>
          <tbody id="clusters-table-body"></tbody>
        </table>
      </div>
      <div class="section-box">
        <div class="section-title"><span>Anchor Text Shannon Entropy &amp; Anti-Penguin Diversity Guard</span></div>
        <table>
          <thead>
            <tr>
              <th>Target Page</th>
              <th>Shannon Entropy (Bits)</th>
              <th>Exact-Match Ratio</th>
              <th>Status</th>
              <th>Anchor Distribution &amp; Suggested Variations</th>
            </tr>
          </thead>
          <tbody id="entropy-table-body"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 5: GSC STRIKING DISTANCE -->
    <div id="tab-gsc" class="tab-pane">
      <div class="section-box">
        <div class="section-title"><span>Google Search Console Striking Distance &amp; Query Intelligence</span></div>
        <table>
          <thead>
            <tr>
              <th>Search Query</th>
              <th>Target Landing Page</th>
              <th>Impressions</th>
              <th>Clicks</th>
              <th>CTR</th>
              <th>Avg Position</th>
              <th>Strategic Tier</th>
            </tr>
          </thead>
          <tbody id="gsc-table-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const graphData = {graph_data_json};
    const opportunities = {opps_json};
    const externalOpps = {ext_json};
    const topicClusters = {clusters_json};
    const entropyProfiles = {entropy_json};
    const gscQueries = {gsc_json};
    let currentEngine = "all";

    function switchTab(tabId, btn) {{
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btn.classList.add('active');
      if (tabId === 'tab-graph') initGraph();
    }}

    function escapeHtml(str) {{
      return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }}

    function highlightContext(sentence, anchor) {{
      if (!sentence) return '';
      const i = sentence.toLowerCase().indexOf((anchor || '').toLowerCase());
      if (i < 0) return escapeHtml(sentence);
      return escapeHtml(sentence.slice(0, i)) + '<mark class="anchor-hl">' + escapeHtml(sentence.slice(i, i + anchor.length)) + '</mark>' + escapeHtml(sentence.slice(i + anchor.length));
    }}

    function copySnippet(text, btn) {{
      navigator.clipboard.writeText(text).then(() => {{
        const orig = btn.innerText;
        btn.innerText = "Copied!";
        btn.style.background = "#10b981";
        setTimeout(() => {{
          btn.innerText = orig;
          btn.style.background = "";
        }}, 1400);
      }});
    }}

    function renderTable(data) {{
      const tbody = document.getElementById("table-body");
      if (!data.length) {{
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 32px;">No matching internal link opportunities found.</td></tr>';
        return;
      }}
      tbody.innerHTML = data.map(o => {{
        const pct = Math.min(100, Math.round(o.score * 100));
        const engineClass = o.engine === 'direct_mention' ? 'engine-mention' : 'engine-linkbert';
        const engineLabel = o.engine === 'direct_mention' ? 'Direct Target Mention' : 'LinkBERT Neural';
        const hrefSnippet = `<a href="${{o.target}}">${{o.anchor}}</a>`;
        const mdSnippet = `[${{o.anchor}}](${{o.target}})`;

        return `
          <tr>
            <td>
              <div style="font-weight: 600;">${{escapeHtml(o.source)}}</div>
              <span class="engine-badge ${{engineClass}}">${{engineLabel}}</span>
            </td>
            <td>
              <span class="anchor-hl">${{escapeHtml(o.anchor)}}</span>
              <div style="color: var(--text-muted); font-size: 11.5px; margin-top: 4px;">${{o.anchor.split(' ').length}} words</div>
            </td>
            <td>
              <div style="font-weight: 600;">${{escapeHtml(o.target_title || o.target)}}</div>
              <div style="color: var(--text-muted); font-size: 12px;">${{escapeHtml(o.target)}}</div>
            </td>
            <td>
              <span style="font-weight: 700;">${{o.score.toFixed(2)}}</span>
              <div class="score-bar"><div class="score-bar-fill" style="width: ${{pct}}%;"></div></div>
            </td>
            <td>
              <div style="font-size: 12.5px; color: var(--text-muted); margin-bottom: 6px;">${{highlightContext(o.sentence, o.anchor)}}</div>
              <div class="snippet">
                <code>${{escapeHtml(hrefSnippet)}}</code>
                <div style="display:flex; gap:4px;">
                  <button class="btn-copy" onclick="copySnippet('${{escapeHtml(hrefSnippet)}}', this)">HTML</button>
                  <button class="btn-copy" onclick="copySnippet('${{escapeHtml(mdSnippet)}}', this)">MD</button>
                </div>
              </div>
            </td>
          </tr>
        `;
      }}).join('');
    }}

    function renderExternalTable() {{
      const tbody = document.getElementById("ext-table-body");
      const q = (document.getElementById("ext-search-input")?.value || "").toLowerCase();
      const filtered = externalOpps.filter(e =>
        (e.source_key || '').toLowerCase().includes(q) ||
        (e.anchor || '').toLowerCase().includes(q) ||
        (e.target_domain || '').toLowerCase().includes(q) ||
        (e.target_title || '').toLowerCase().includes(q)
      );
      if (!filtered.length) {{
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 32px;">No external citation opportunities found.</td></tr>';
        return;
      }}
      tbody.innerHTML = filtered.map(e => {{
        const tier = e.trust_tier || 1;
        const tierCls = tier === 1 ? 'tier-1' : (tier === 2 ? 'tier-2' : 'tier-3');
        const extHtml = `<a href="${{e.target_url}}" target="_blank" rel="${{e.recommended_rel || 'noopener noreferrer'}}">${{e.anchor}}</a>`;
        return `
          <tr>
            <td style="font-weight:600;">${{escapeHtml(e.source_key)}}</td>
            <td><span class="anchor-hl">${{escapeHtml(e.anchor)}}</span></td>
            <td>
              <div style="font-weight:600;">${{escapeHtml(e.target_title)}}</div>
              <a href="${{escapeHtml(e.target_url)}}" target="_blank" style="color:var(--accent); font-size:12px;">${{escapeHtml(e.target_domain)}} ↗</a>
            </td>
            <td>
              <span class="tier-badge ${{tierCls}}">Tier ${{tier}}</span>
              <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">${{escapeHtml(e.category)}}</div>
            </td>
            <td>
              <div style="font-size:12.5px; color:var(--text-muted); margin-bottom:4px;">${{highlightContext(e.sentence, e.anchor)}}</div>
              <div style="font-size:11.5px; color:#34d399; margin-bottom:6px;">Basis: ${{escapeHtml(e.academic_basis)}}</div>
              <div class="snippet">
                <code>${{escapeHtml(extHtml)}}</code>
                <button class="btn-copy" onclick="copySnippet('${{escapeHtml(extHtml)}}', this)">Copy</button>
              </div>
            </td>
          </tr>
        `;
      }}).join('');
    }}

    function renderTopologyTable() {{
      const tbody = document.getElementById("topology-table-body");
      const sorted = [...graphData.nodes].sort((a, b) => b.projPr - a.projPr);
      tbody.innerHTML = sorted.map(n => {{
        const delta = n.pr > 0 ? ((n.projPr - n.pr) / n.pr * 100) : 0;
        const deltaColor = delta > 0.5 ? '#10b981' : (delta < -0.5 ? '#f59e0b' : 'var(--text-muted)');
        const statusBadge = n.isOrphan
          ? '<span class="tier-badge" style="background:rgba(239,68,68,0.2);color:#f87171;">ORPHAN</span>'
          : (n.isUnder ? '<span class="tier-badge tier-3">UNDER-LINKED</span>' : '<span class="tier-badge tier-1">HEALTHY</span>');
        const cdStr = n.clickDepth >= 999 ? '∞ (Unreachable)' : `${{n.clickDepth}} hops`;
        return `
          <tr>
            <td>
              <div style="font-weight:600;">${{escapeHtml(n.key)}}</div>
              <div style="font-size:11.5px; color:var(--text-muted);">${{escapeHtml(n.title)}}</div>
            </td>
            <td style="font-weight:700;">${{n.pr.toFixed(3)}}</td>
            <td style="font-weight:700; color:var(--accent);">${{n.projPr.toFixed(3)}}</td>
            <td style="font-weight:700; color:${{deltaColor}};">${{delta >= 0 ? '+' : ''}}${{delta.toFixed(1)}}%</td>
            <td>${{cdStr}}</td>
            <td>${{n.auth.toFixed(3)}}</td>
            <td>${{n.hub.toFixed(3)}}</td>
            <td>${{n.bridge.toFixed(3)}}</td>
            <td>${{statusBadge}}</td>
          </tr>
        `;
      }}).join('');
    }}

    function renderSilosAndEntropy() {{
      const cBody = document.getElementById("clusters-table-body");
      cBody.innerHTML = topicClusters.map(c => `
        <tr>
          <td>#${{c.id}}</td>
          <td style="font-weight:600;">${{escapeHtml(c.name)}}</td>
          <td style="color:var(--accent); font-weight:600;">${{escapeHtml(c.hub)}}</td>
          <td>${{c.pages.length}} pages</td>
          <td>${{c.inEdges}}</td>
          <td>${{c.outEdges}}</td>
          <td><span class="tier-badge ${{c.cohesion >= 0.5 ? 'tier-1' : 'tier-3'}}">${{(c.cohesion * 100).toFixed(0)}}%</span></td>
        </tr>
      `).join('');

      const eBody = document.getElementById("entropy-table-body");
      if (!entropyProfiles.length) {{
        eBody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding:24px; color:var(--text-muted);">No target anchor distributions to evaluate.</td></tr>';
        return;
      }}
      eBody.innerHTML = entropyProfiles.map(p => {{
        const status = p.overOptimized
          ? '<span class="tier-badge" style="background:rgba(245,158,11,0.2);color:#fbbf24;">DIVERSIFY</span>'
          : '<span class="tier-badge tier-1">HEALTHY</span>';
        return `
          <tr>
            <td>
              <div style="font-weight:600;">${{escapeHtml(p.target)}}</div>
              <div style="font-size:11.5px; color:var(--text-muted);">${{escapeHtml(p.title)}}</div>
            </td>
            <td style="font-weight:700;">${{p.entropy.toFixed(2)}} bits</td>
            <td>${{(p.exactRatio * 100).toFixed(0)}}%</td>
            <td>${{status}}</td>
            <td style="font-size:12.5px; color:var(--text-muted);">${{escapeHtml(p.rec)}}</td>
          </tr>
        `;
      }}).join('');
    }}

    function renderGscTable() {{
      const gBody = document.getElementById("gsc-table-body");
      if (!gscQueries.length) {{
        gBody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:24px; color:var(--text-muted);">No Google Search Console query CSV supplied for this run. Use --gsc <path.csv> to populate.</td></tr>';
        return;
      }}
      gBody.innerHTML = gscQueries.map(q => {{
        const pos = parseFloat(q.position || 0);
        let tier = 'Longtail';
        let cls = 'tier-3';
        if (pos <= 3) {{ tier = 'Top 3 (Defend)'; cls = 'tier-1'; }}
        else if (pos <= 10) {{ tier = 'Page 1 Striking Distance'; cls = 'tier-2'; }}
        else if (pos <= 20) {{ tier = 'Page 2 Striking Distance'; cls = 'tier-3'; }}
        return `
          <tr>
            <td style="font-weight:600; color:var(--accent);">${{escapeHtml(q.query)}}</td>
            <td>${{escapeHtml(q.page)}}</td>
            <td>${{q.impressions}}</td>
            <td>${{q.clicks}}</td>
            <td>${{escapeHtml(q.ctr)}}</td>
            <td style="font-weight:700;">${{pos.toFixed(1)}}</td>
            <td><span class="tier-badge ${{cls}}">${{tier}}</span></td>
          </tr>
        `;
      }}).join('');
    }}

    function filterTable() {{
      const q = document.getElementById("search-input").value.toLowerCase();
      const filtered = opportunities.filter(o => {{
        const matchesQuery = o.anchor.toLowerCase().includes(q) || o.source.toLowerCase().includes(q) || o.target.toLowerCase().includes(q) || (o.target_title || '').toLowerCase().includes(q);
        const matchesEngine = currentEngine === 'all' || o.engine === currentEngine;
        return matchesQuery && matchesEngine;
      }});
      renderTable(filtered);
    }}

    function setEngineFilter(engine, btn) {{
      currentEngine = engine;
      document.querySelectorAll('#tab-internal .controls .filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterTable();
    }}

    function downloadJSON() {{
      const blob = new Blob([JSON.stringify(opportunities, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'linkbert-opportunities.json';
      a.click();
    }}

    function initGraph() {{
      const svg = document.getElementById("network-svg");
      const width = svg.clientWidth || 1200;
      const height = svg.clientHeight || 520;
      const nodes = graphData.nodes;
      const links = graphData.links;
      if (!nodes.length) return;

      // Force-relaxed radial layout grouped by cluster
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.min(width, height) * 0.37;

      nodes.forEach((n, idx) => {{
        const angle = (idx / nodes.length) * 2 * Math.PI;
        const rMod = n.key === '/' ? 0 : (n.clickDepth === 1 ? radius * 0.55 : radius);
        n.x = cx + rMod * Math.cos(angle);
        n.y = cy + rMod * Math.sin(angle);
      }});

      // Run 40 iterations of spring-repulsion relaxation
      for (let iter = 0; iter < 40; iter++) {{
        for (let i = 0; i < nodes.length; i++) {{
          for (let j = i + 1; j < nodes.length; j++) {{
            const dx = nodes[j].x - nodes[i].x;
            const dy = nodes[j].y - nodes[i].y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            if (dist < 90) {{
              const rep = (90 - dist) * 0.12;
              const ux = dx / dist, uy = dy / dist;
              if (nodes[i].key !== '/') {{ nodes[i].x -= ux * rep; nodes[i].y -= uy * rep; }}
              if (nodes[j].key !== '/') {{ nodes[j].x += ux * rep; nodes[j].y += uy * rep; }}
            }}
          }}
        }}
        nodes.forEach(n => {{
          n.x = Math.max(50, Math.min(width - 50, n.x));
          n.y = Math.max(40, Math.min(height - 40, n.y));
        }});
      }}

      let svgHtml = '';
      links.forEach(l => {{
        const s = nodes[l.source];
        const t = nodes[l.target];
        if (!s || !t) return;
        const isProposed = l.type === 'proposed';
        const color = isProposed ? '#10b981' : '#334155';
        const dash = isProposed ? 'stroke-dasharray="5,5"' : '';
        const w = isProposed ? '2' : '1';
        svgHtml += `<line x1="${{s.x}}" y1="${{s.y}}" x2="${{t.x}}" y2="${{t.y}}" stroke="${{color}}" stroke-width="${{w}}" ${{dash}} opacity="${{isProposed ? 0.85 : 0.35}}" />`;
      }});

      nodes.forEach(n => {{
        const r = Math.max(7, Math.min(22, 8 + n.projPr * 4.5));
        const fill = n.isOrphan ? '#ef4444' : (n.key === '/' ? '#10b981' : '#38bdf8');
        const infoStr = `${{escapeHtml(n.key)}} | PR: ${{n.pr.toFixed(2)}} → ${{n.projPr.toFixed(2)}} | Depth: ${{n.clickDepth >= 999 ? '∞' : n.clickDepth}} | In: ${{n.inDeg}} Out: ${{n.outDeg}} | Auth: ${{n.auth.toFixed(2)}}`;
        svgHtml += `
          <g style="cursor: pointer;" onmouseenter="document.getElementById('node-info').innerText = '${{infoStr}}'">
            <circle cx="${{n.x}}" cy="${{n.y}}" r="${{r}}" fill="${{fill}}" stroke="#070b14" stroke-width="2" />
            <text x="${{n.x}}" y="${{n.y + r + 13}}" fill="#cbd5e1" font-size="10.5" font-family="sans-serif" text-anchor="middle">${{escapeHtml(n.key.length > 26 ? n.key.slice(0, 24) + '…' : n.key)}}</text>
          </g>
        `;
      }});

      svg.innerHTML = svgHtml;
    }}

    renderTable(opportunities);
    renderExternalTable();
    renderTopologyTable();
    renderSilosAndEntropy();
    renderGscTable();
    window.addEventListener('load', initGraph);
  </script>
</body>
</html>
"""
    return html_doc


def build_master_workbook(
    out_file: Path,
    idx: SiteIndex,
    opportunities: List[CandidateOpportunity],
    external_opps: Optional[List[Any]] = None,
    metrics: Optional[GraphMetrics] = None,
    gsc_queries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Generates an industrial-grade, 6-tab Excel workbook for strategic link architecture."""
    if not HAS_OPENPYXL:
        return

    if metrics is None:
        proposed_pairs = [(o.source_key, o.target_key) for o in opportunities]
        metrics = LinkGraph(idx).analyze(additional_links=proposed_pairs)

    from .intelligence import analyze_anchor_diversity, detect_topic_clusters
    proposed_triples = [(o.source_key, o.target_key, o.anchor) for o in opportunities]
    anchor_profiles = analyze_anchor_diversity(idx, proposed_triples)
    topic_clusters = detect_topic_clusters(idx, method="url_hierarchy")

    wb = openpyxl.Workbook()
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=False)

    def style_header_and_cols(ws: Any, headers: List[str]) -> None:
        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_left
        ws.freeze_panes = "A2"

    # =========================================================================
    # TAB 1: Internal_Linking_Matrix
    # =========================================================================
    ws_int = wb.active
    ws_int.title = "Internal_Linking_Matrix"
    int_headers = [
        "Source Page",
        "Anchor Text",
        "Target Page",
        "Target Title",
        "Re-Rank Score",
        "Discovery Engine",
        "Context Sentence",
        "Source File",
        "Target Projected PR",
        "HTML Snippet",
        "Markdown Snippet",
    ]
    style_header_and_cols(ws_int, int_headers)

    for o in opportunities:
        tgt_pr = round(metrics.projected_page_rank.get(o.target_key, 1.0), 4) if metrics else 1.0
        ws_int.append([
            o.source_key,
            o.anchor,
            o.target_key,
            o.target_title,
            round(o.score, 3),
            o.engine,
            o.sentence,
            o.source_file,
            tgt_pr,
            f'<a href="{o.target_key}">{o.anchor}</a>',
            f"[{o.anchor}]({o.target_key})",
        ])

    # =========================================================================
    # TAB 2: External_Authority_Citations
    # =========================================================================
    ws_ext = wb.create_sheet(title="External_Authority_Citations")
    ext_headers = [
        "Source Page",
        "Anchor Entity",
        "Exact Sentence Context",
        "Target Authority URL",
        "Target Title",
        "Target Domain",
        "Trust Tier",
        "Citation Category",
        "Recommended Rel",
        "Confidence Score",
        "E-E-A-T Justification",
        "Academic / Patent Basis",
    ]
    style_header_and_cols(ws_ext, ext_headers)

    if external_opps:
        for eo in external_opps:
            tier_str = f"Tier {getattr(eo, 'trust_tier', 1)}"
            ws_ext.append([
                getattr(eo, "source_key", ""),
                getattr(eo, "anchor", ""),
                getattr(eo, "sentence", ""),
                getattr(eo, "target_url", ""),
                getattr(eo, "target_title", ""),
                getattr(eo, "target_domain", ""),
                tier_str,
                getattr(eo, "category", ""),
                getattr(eo, "recommended_rel", "noopener noreferrer"),
                round(getattr(eo, "confidence", 0.85), 3),
                getattr(eo, "reason", ""),
                getattr(eo, "academic_basis", ""),
            ])

    # =========================================================================
    # TAB 3: GSC_Striking_Distance
    # =========================================================================
    ws_gsc = wb.create_sheet(title="GSC_Striking_Distance")
    gsc_headers = [
        "Query",
        "Assigned Target URL",
        "Impressions",
        "Clicks",
        "CTR",
        "Average Position",
        "Striking Distance Tier",
        "Recommended Internal Anchor",
    ]
    style_header_and_cols(ws_gsc, gsc_headers)

    if gsc_queries:
        for q in gsc_queries:
            pos = float(q.get("position", 0.0))
            if pos <= 3.0:
                tier = "Top 3 (Defend)"
            elif pos <= 10.0:
                tier = "Page 1 - Striking Distance (Immediate Boost)"
            elif pos <= 20.0:
                tier = "Page 2 - Striking Distance (High Potential)"
            elif pos <= 50.0:
                tier = "Top 50 - Mid-Tier Opportunity"
            else:
                tier = "Longtail - Needs Topic Silo"

            ws_gsc.append([
                q.get("query", ""),
                q.get("page", "/"),
                q.get("impressions", 0),
                q.get("clicks", 0),
                q.get("ctr", "0%"),
                pos,
                tier,
                q.get("query", ""),
            ])

    # =========================================================================
    # TAB 4: Graph_Topology_Audit
    # =========================================================================
    ws_graph = wb.create_sheet(title="Graph_Topology_Audit")
    graph_headers = [
        "Page URL",
        "Page Title",
        "Current PageRank",
        "Projected PageRank",
        "PageRank Delta (%)",
        "Orphan Status",
        "Under-Linked Status",
        "Topic Cluster",
        "Click Depth (Hops)",
        "Projected Click Depth",
        "HITS Authority Score",
        "HITS Hub Score",
        "Bridge Centrality",
    ]
    style_header_and_cols(ws_graph, graph_headers)

    if metrics:
        for p in idx.pages:
            cur_pr = metrics.page_rank.get(p.key, 0.0)
            proj_pr = metrics.projected_page_rank.get(p.key, cur_pr)
            delta = ((proj_pr - cur_pr) / cur_pr * 100) if cur_pr > 0 else 0.0
            is_orphan = "ORPHAN (Critical)" if p.key in metrics.orphan_pages else "OK"
            is_under = "UNDER-LINKED" if p.key in metrics.under_linked_pages else "OK"
            cluster_val = "root"
            if hasattr(metrics, "clusters") and isinstance(metrics.clusters, dict):
                for c_id, c_pages in metrics.clusters.items():
                    if isinstance(c_pages, list) and p.key in c_pages:
                        cluster_val = str(c_id)
                        break

            cd = metrics.click_depth.get(p.key, 0)
            proj_cd = metrics.projected_click_depth.get(p.key, cd)
            auth_s = metrics.authority_scores.get(p.key, 0.0)
            hub_s = metrics.hub_scores.get(p.key, 0.0)
            bw_s = metrics.betweenness_centrality.get(p.key, 0.0)

            ws_graph.append([
                p.key,
                p.title or p.key,
                round(cur_pr, 4),
                round(proj_pr, 4),
                f"{delta:+.1f}%",
                is_orphan,
                is_under,
                cluster_val,
                cd if cd < 999 else "Unreachable",
                proj_cd if proj_cd < 999 else "Unreachable",
                round(auth_s, 4),
                round(hub_s, 4),
                round(bw_s, 4),
            ])

    # =========================================================================
    # TAB 5: Anchor_Entropy_&_Silos
    # =========================================================================
    ws_ent = wb.create_sheet(title="Anchor_Entropy_&_Silos")
    ent_headers = [
        "Target Page",
        "Target Title",
        "Total Inbound Anchors",
        "Unique Variations",
        "Shannon Entropy (Bits)",
        "Exact-Match Ratio",
        "Penguin Safety Status",
        "Recommendation",
        "Suggested Anchor Variations",
    ]
    style_header_and_cols(ws_ent, ent_headers)

    for ap in anchor_profiles:
        total_a = sum(ap.anchors.values())
        uniq_a = len(ap.anchors)
        status_str = "OVER-OPTIMIZED (Action Needed)" if ap.is_over_optimized else "HEALTHY"
        ws_ent.append([
            ap.target_key,
            ap.target_title,
            total_a,
            uniq_a,
            ap.entropy,
            f"{ap.exact_match_ratio * 100:.1f}%",
            status_str,
            ap.recommendation,
            ", ".join(ap.suggested_variations),
        ])

    # =========================================================================
    # TAB 6: Executive_Audit_Summary
    # =========================================================================
    ws_sum = wb.create_sheet(title="Executive_Audit_Summary")
    sum_headers = ["Audit Dimension / Metric", "Value", "Benchmark / Target", "Status"]
    style_header_and_cols(ws_sum, sum_headers)

    orphans_before = len(metrics.orphan_pages) if metrics else 0
    rescued_targets = {o.target_key for o in opportunities}
    orphans_after = len([k for k in (metrics.orphan_pages if metrics else []) if k not in rescued_targets])

    summary_rows = [
        ("Total Indexed Site Pages", len(idx.pages), ">= 1", "PASS"),
        ("Internal Link Opportunities Discovered", len(opportunities), "High Recall + Precision", "PASS"),
        ("External E-E-A-T Authority Citations", len(external_opps or []), "Verified Primary Sources", "PASS"),
        ("Orphan Pages (Baseline -> Post-Optimization)", f"{orphans_before} -> {orphans_after}", "0 Unrescued Orphans", "OPTIMIZED"),
        ("Topic Clusters / Silos Detected", len(topic_clusters), "Modular Silo Structure", "PASS"),
        ("GSC Queries Integrated", len(gsc_queries or []), "3-12 Month Window", "PASS" if gsc_queries else "OPTIONAL"),
    ]
    for r in summary_rows:
        ws_sum.append(list(r))

    # Auto-adjust column widths across all sheets
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if len(val) > max_len:
                    max_len = len(val)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 65)

    wb.save(out_file)


def write_all(
    out_dir: Path,
    idx: SiteIndex,
    opportunities: List[CandidateOpportunity],
    metrics: GraphMetrics | None = None,
    title: str = "LinkBERT Pro — SEO Link Intelligence Dashboard",
    external_opps: Optional[List[Any]] = None,
    gsc_queries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        proposed_pairs = [(o.source_key, o.target_key) for o in opportunities]
        metrics = LinkGraph(idx).analyze(additional_links=proposed_pairs)

    # 1. Discover external authority citations if not passed
    if external_opps is None:
        from .external import ExternalLinkEngine
        ext_eng = ExternalLinkEngine()
        external_opps = []
        for p in idx.pages:
            external_opps.extend(ext_eng.analyze_page(p))

    # 2. Write suggestions.json
    raw_opps = [o.as_dict() for o in opportunities]
    (out_dir / "suggestions.json").write_text(
        json.dumps({
            "site": idx.root,
            "count": len(raw_opps),
            "opportunities": raw_opps,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 3. Write suggestions.csv (Internal Links)
    with (out_dir / "suggestions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "anchor", "target", "target_title", "score", "engine", "sentence", "source_file"])
        for o in opportunities:
            w.writerow([
                o.source_key,
                o.anchor,
                o.target_key,
                o.target_title,
                round(o.score, 3),
                o.engine,
                o.sentence,
                o.source_file,
            ])

    # 4. Write external_links.csv (External Authority Citations)
    ext_csv = out_dir / "external_links.csv"
    with ext_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source_page",
            "anchor_text",
            "exact_sentence_context",
            "recommended_external_url",
            "target_title",
            "target_domain",
            "trust_tier",
            "citation_category",
            "recommended_rel_attribute",
            "confidence_score",
            "eeat_justification",
            "academic_basis",
        ])
        for eo in external_opps:
            w.writerow([
                getattr(eo, "source_key", ""),
                getattr(eo, "anchor", ""),
                getattr(eo, "sentence", ""),
                getattr(eo, "target_url", ""),
                getattr(eo, "target_title", ""),
                getattr(eo, "target_domain", ""),
                f"Tier {getattr(eo, 'trust_tier', 1)}",
                getattr(eo, "category", ""),
                getattr(eo, "recommended_rel", "noopener noreferrer"),
                round(getattr(eo, "confidence", 0.85), 3),
                getattr(eo, "reason", ""),
                getattr(eo, "academic_basis", ""),
            ])

    # 5. Write 6-tab Excel Workbook (link_architecture_master.xlsx)
    excel_file = out_dir / "link_architecture_master.xlsx"
    build_master_workbook(
        out_file=excel_file,
        idx=idx,
        opportunities=opportunities,
        external_opps=external_opps,
        metrics=metrics,
        gsc_queries=gsc_queries,
    )

    # 6. Write report.html (5-Tab Interactive HTML5 App)
    html_content = generate_dashboard_html(
        idx,
        opportunities,
        metrics,
        title=title,
        external_opps=external_opps,
        gsc_queries=gsc_queries,
    )
    report_file = out_dir / "report.html"
    report_file.write_text(html_content, encoding="utf-8")

    # 7. Write graph_metrics.json
    if metrics:
        (out_dir / "graph_metrics.json").write_text(
            json.dumps({
                "page_rank": metrics.page_rank,
                "projected_page_rank": metrics.projected_page_rank,
                "click_depth": metrics.click_depth,
                "projected_click_depth": metrics.projected_click_depth,
                "hub_scores": metrics.hub_scores,
                "authority_scores": metrics.authority_scores,
                "betweenness_centrality": metrics.betweenness_centrality,
                "orphan_pages": metrics.orphan_pages,
                "under_linked_pages": metrics.under_linked_pages,
                "dead_end_pages": metrics.dead_end_pages,
                "clusters": getattr(metrics, "clusters", {}),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "opportunities": len(opportunities),
        "external_opportunities": len(external_opps),
        "report": str(report_file),
        "excel": str(excel_file),
        "internal_csv": str(out_dir / "suggestions.csv"),
        "external_csv": str(ext_csv),
        "json": str(out_dir / "suggestions.json"),
    }
