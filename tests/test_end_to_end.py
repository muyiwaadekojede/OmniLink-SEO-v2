"""Comprehensive automated verification suite (24 tests) for LinkBERT Pro."""

import os
import shutil
import unittest
from pathlib import Path

from lb.apply import LinkApplier
from lb.discovery import BM25FIndex, HybridDiscoveryEngine, check_topical_coherence
from lb.external import ExternalLinkEngine, resolve_empirical_target
from lb.extract import parse_html, parse_jsx, parse_markdown
from lb.graph import LinkGraph
from lb.model import _snap_word_boundaries, _trim
from lb.site import build


class TestLinkBertPro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.fixture_dir = cls.root / "fixture"
        cls.test_index = cls.root / ".test_index"
        try:
            import sentence_transformers
            has_embed = True
        except ImportError:
            has_embed = False
        cls.idx = build(str(cls.fixture_dir), cls.test_index, embed=has_embed)

    @classmethod
    def tearDownClass(cls):
        if cls.test_index.exists():
            shutil.rmtree(cls.test_index)

    def test_01_indexing_and_pages(self):
        self.assertEqual(len(self.idx.pages), 6)
        keys = {p.key for p in self.idx.pages}
        self.assertIn("/", keys)
        self.assertIn("/senior-pets/", keys)
        self.assertIn("/accident-only-cover/", keys)
        self.assertIn("/routine-care/", keys)
        self.assertIn("/compare-pet-insurance/", keys)
        self.assertIn("/blog/choosing-a-vet/", keys)

    def test_02_graph_and_pagerank(self):
        graph = LinkGraph(self.idx)
        metrics = graph.analyze()
        self.assertIn("/compare-pet-insurance/", metrics.page_rank)
        self.assertGreater(metrics.page_rank["/compare-pet-insurance/"], 1.0)
        self.assertEqual(len(metrics.orphan_pages), 4)

    def test_03_hybrid_discovery(self):
        engine = HybridDiscoveryEngine(self.idx, predictor=None)
        senior_page = self.idx.by_key["/senior-pets/"]
        opps = engine.find_direct_mentions(senior_page, set())

        targets = {o.target_key for o in opps}
        self.assertIn("/accident-only-cover/", targets)
        self.assertIn("/routine-care/", targets)

    def test_04_html_apply_and_rollback(self):
        temp_dir = self.root / "temp_apply_test"
        shutil.copytree(self.fixture_dir, temp_dir)

        try:
            temp_idx = build(str(temp_dir), temp_dir / ".idx", embed=False)
            engine = HybridDiscoveryEngine(temp_idx, predictor=None)
            senior_page = temp_idx.by_key["/senior-pets/"]
            opps = engine.find_direct_mentions(senior_page, set())

            applier = LinkApplier(dry_run=False, create_backup=True)
            res = applier.apply({senior_page.source: opps})
            self.assertEqual(len(res), 1)
            self.assertGreater(res[0].applied_count, 0)

            modified_text = Path(senior_page.source).read_text(encoding="utf-8")
            self.assertIn('<a href="/accident-only-cover/">', modified_text)

            restored = LinkApplier.rollback(temp_dir)
            self.assertGreater(len(restored), 0)
            restored_text = Path(senior_page.source).read_text(encoding="utf-8")
            self.assertNotIn('<a href="/accident-only-cover/">', restored_text)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_05_external_engine(self):
        ext = ExternalLinkEngine()
        home = self.idx.by_key["/"]
        opps = ext.analyze_page(home)
        self.assertIsInstance(opps, list)

    def test_06_topic_clusters(self):
        from lb.intelligence import detect_topic_clusters
        clusters_hier = detect_topic_clusters(self.idx, method="url_hierarchy")
        self.assertGreaterEqual(len(clusters_hier), 4)

        clusters_louvain = detect_topic_clusters(self.idx, method="louvain")
        self.assertGreaterEqual(len(clusters_louvain), 4)
        for c in clusters_louvain:
            self.assertTrue(c.hub_page in self.idx.by_key)

    def test_07_anchor_diversity(self):
        from lb.intelligence import analyze_anchor_diversity
        proposed = [
            ("/senior-pets/", "/accident-only-cover/", "accident-only cover"),
            ("/", "/accident-only-cover/", "accident-only cover"),
            ("/blog/choosing-a-vet/", "/accident-only-cover/", "accident-only cover"),
        ]
        profiles = analyze_anchor_diversity(self.idx, proposed)
        self.assertGreater(len(profiles), 0)
        profile_acc = next(p for p in profiles if p.target_key == "/accident-only-cover/")
        self.assertEqual(profile_acc.entropy, 0.0)
        self.assertTrue(profile_acc.is_over_optimized)
        self.assertGreater(len(profile_acc.suggested_variations), 0)

    def test_08_markdown_apply_and_rollback(self):
        temp_md = self.root / "temp_test.md"
        temp_md.write_text("# Guide\n\nFor more details check accident-only cover here.\n", encoding="utf-8")
        try:
            senior_page = self.idx.by_key["/senior-pets/"]
            engine = HybridDiscoveryEngine(self.idx, predictor=None)
            opps = engine.find_direct_mentions(senior_page, set())
            app_opps = [o for o in opps if o.target_key == "/accident-only-cover/"]

            applier = LinkApplier(dry_run=False, create_backup=True)
            res = applier.apply_to_markdown(temp_md, app_opps)
            self.assertEqual(res.applied_count, 1)
            modified = temp_md.read_text(encoding="utf-8")
            self.assertIn("[accident-only cover](/accident-only-cover/)", modified)

            restored = LinkApplier.rollback(self.root)
            self.assertIn(str(temp_md), restored)
            rolled_back = temp_md.read_text(encoding="utf-8")
            self.assertNotIn("](/accident-only-cover/)", rolled_back)
        finally:
            if temp_md.exists():
                temp_md.unlink()
            bak = temp_md.with_suffix(temp_md.suffix + ".bak")
            if bak.exists():
                bak.unlink()

    def test_09_cannibalization_detection(self):
        from lb.browser import parse_search_console_csv
        from lb.intelligence import detect_cannibalization
        csv_text = (
            "Top queries,Clicks,Impressions,CTR,Position,Top pages\n"
            "pet insurance,100,2000,5.0%,3.0,https://example.com/page1\n"
            "pet insurance,50,1500,3.3%,7.5,https://example.com/page2\n"
        )
        rows = parse_search_console_csv(csv_text)
        self.assertEqual(len(rows), 2)
        issues = detect_cannibalization(rows, min_impressions=10)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].query, "pet insurance")
        self.assertIn("https://example.com/page1", issues[0].recommendation)
        self.assertEqual(issues[0].wasted_impressions, 1500)

    def test_10_serp_conflict_guard(self):
        from lb.browser import SERPConflict, SERPResult, SearchConsoleRow, check_serp_conflicts
        ext_domains = {"competitor.com", "gov.au"}
        sc_rows = [SearchConsoleRow(query="pet insurance", page="/", clicks=10, impressions=100, ctr=0.1, position=3.0)]
        serp = {
            "pet insurance": [
                SERPResult(title="Competitor", url="https://competitor.com/best-pet-insurance", domain="competitor.com", position=4, snippet="..."),
                SERPResult(title="Wiki", url="https://en.wikipedia.org/wiki/Pet_insurance", domain="wikipedia.org", position=1, snippet="..."),
            ]
        }
        conflicts = check_serp_conflicts(ext_domains, sc_rows, serp)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].external_domain, "competitor.com")
        self.assertEqual(conflicts[0].risk_level, "high")

    def test_11_external_link_engine(self):
        from lb.external import ExternalLinkEngine
        ext_engine = ExternalLinkEngine()
        all_ext = []
        for p in self.idx.pages:
            all_ext.extend(ext_engine.analyze_page(p))
        self.assertIsInstance(all_ext, list)
        for o in all_ext:
            self.assertIn(o.trust_tier, (1, 2, 3, 4))
            self.assertTrue(len(o.anchor) > 1)
            self.assertIn("noopener", o.recommended_rel)

    def test_12_multi_tab_excel_export(self):
        import openpyxl
        from lb.report import write_all
        eng = HybridDiscoveryEngine(self.idx)
        senior_page = self.idx.by_key["/senior-pets/"]
        opps = eng.find_direct_mentions(senior_page, set())
        stats = write_all(self.test_index, self.idx, opps)
        self.assertIn("excel", stats)
        wb = openpyxl.load_workbook(stats["excel"], read_only=True)
        self.assertIn("Internal_Linking_Matrix", wb.sheetnames)
        self.assertIn("External_Authority_Citations", wb.sheetnames)
        self.assertIn("GSC_Striking_Distance", wb.sheetnames)
        self.assertIn("Graph_Topology_Audit", wb.sheetnames)
        self.assertIn("Anchor_Entropy_&_Silos", wb.sheetnames)
        self.assertIn("Executive_Audit_Summary", wb.sheetnames)
        wb.close()

    def test_13_specialist_audit_gate(self):
        from tests.verify_deliverable import audit_workbook
        stats = audit_workbook(self.test_index / "link_architecture_master.xlsx")
        self.assertEqual(stats["status"], "PASS")
        self.assertEqual(stats["total_defects"], 0)
        self.assertEqual(stats["specialists_active"], 6)

    def test_14_kleinberg_hits_and_click_depth(self):
        graph = LinkGraph(self.idx)
        metrics = graph.analyze(additional_links=[("/compare-pet-insurance/", "/senior-pets/")])
        # Homepage has click depth 0
        self.assertEqual(metrics.click_depth["/"], 0)
        # Compare Pet Insurance is linked from '/', so click depth is 1
        self.assertEqual(metrics.click_depth["/compare-pet-insurance/"], 1)
        # Senior pets was orphan (999), after proposed link from compare-pet-insurance (depth 1) its projected depth is 2
        self.assertEqual(metrics.click_depth["/senior-pets/"], 999)
        self.assertEqual(metrics.projected_click_depth["/senior-pets/"], 2)
        # Kleinberg HITS scores exist and are in [0, 1]
        self.assertIn("/", metrics.hub_scores)
        self.assertIn("/compare-pet-insurance/", metrics.authority_scores)
        self.assertGreater(metrics.authority_scores["/compare-pet-insurance/"], 0.0)

    def test_15_topic_sensitive_pagerank_haveliwala(self):
        from lb.intelligence import detect_topic_clusters
        clusters = detect_topic_clusters(self.idx, method="hybrid")
        self.assertGreater(len(clusters), 0)
        for c in clusters:
            self.assertGreater(len(c.topical_pagerank), 0)
            for pk in c.pages:
                self.assertIn(pk, c.topical_pagerank)
                self.assertGreater(c.topical_pagerank[pk], 0.0)

    def test_16_bm25f_field_weighted_retrieval(self):
        bm25 = BM25FIndex(self.idx)
        scores = bm25.score_query("accident only cover emergency vet")
        self.assertEqual(len(scores), len(self.idx.pages))
        best_idx = int(scores.argmax())
        self.assertEqual(self.idx.pages[best_idx].key, "/accident-only-cover/")
        self.assertAlmostEqual(float(scores.max()), 1.0, places=4)

    def test_17_markdown_frontmatter_and_relative_links(self):
        md_text = (
            "---\n"
            "title: \"Ultimate Technical SEO Guide\"\n"
            "description: \"Complete guide to Core Web Vitals and crawlability\"\n"
            "---\n\n"
            "# Technical SEO Architecture\n\n"
            "Read our [accident-only coverage guide](/accident-only-cover/) and "
            "official [W3C specs](https://www.w3.org/) for full compliance.\n"
        )
        doc = parse_markdown(md_text)
        self.assertEqual(doc.title, "Ultimate Technical SEO Guide")
        self.assertEqual(doc.description, "Complete guide to Core Web Vitals and crawlability")
        self.assertIn("/accident-only-cover/", doc.links)
        self.assertIn("https://www.w3.org/", doc.links)
        self.assertEqual(len(doc.link_details), 2)
        self.assertEqual(doc.link_details[0].anchor, "accident-only coverage guide")
        self.assertFalse(doc.link_details[0].is_external)
        self.assertTrue(doc.link_details[1].is_external)

    def test_18_jsx_tsx_nextjs_component_extraction(self):
        tsx_code = """
        import Link from 'next/link';
        export const metadata = {
          title: 'Next.js International SEO Case Studies',
          description: 'How we scaled organic traffic across 12 markets.'
        };
        export default function Page() {
          return (
            <main>
              <h1>International SEO Growth Results</h1>
              <p>We resolved client-side rendering barriers and connected <Link href="/tools/pagespeed-compare">Core Web Vitals telemetry</Link> across all regional hubs.</p>
            </main>
          );
        }
        """
        doc = parse_jsx(tsx_code)
        self.assertEqual(doc.title, "Next.js International SEO Case Studies")
        self.assertEqual(doc.h1, "International SEO Growth Results")
        self.assertIn("/tools/pagespeed-compare", doc.links)
        self.assertEqual(doc.link_details[0].anchor, "Core Web Vitals telemetry")
        self.assertGreaterEqual(len(doc.blocks), 2)

    def test_19_gsc_striking_distance_injection(self):
        gsc_queries = [
            {
                "query": "older dog insurance options",
                "page": "/routine-care/",
                "clicks": 15,
                "impressions": 800,
                "ctr": "1.9%",
                "position": 11.2,
            }
        ]
        eng = HybridDiscoveryEngine(self.idx, predictor=None, gsc_queries=gsc_queries)
        aliases = dict(eng.registry.target_phrases.get("/routine-care/", []))
        self.assertIn("older dog insurance options", aliases)
        self.assertGreaterEqual(aliases["older dog insurance options"], 0.95)

    def test_20_multi_industry_external_citations(self):
        # E-Commerce Cart Abandonment -> Baymard
        res_ecom = resolve_empirical_target("70% of online shoppers", "Studies show 70% of online shoppers abandon their checkout during cart abandonment.")
        self.assertIsNotNone(res_ecom)
        self.assertEqual(res_ecom[2], "baymard.com")

        # Veterinary -> AVMA
        res_vet = resolve_empirical_target("65% of senior pets", "Over 65% of senior pets require annual veterinary dental care.")
        self.assertIsNotNone(res_vet)
        self.assertEqual(res_vet[2], "avma.org")

        # Cybersecurity -> OWASP
        res_sec = resolve_empirical_target("40% of data breaches", "Nearly 40% of data breaches involve a web application vulnerability or sql injection.")
        self.assertIsNotNone(res_sec)
        self.assertEqual(res_sec[2], "owasp.org")

    def test_21_compound_word_and_numeric_boundary_snapping(self):
        txt = "We lost 1,212 backlinks during the accident-only migration (which hurt rankings"
        # Slicing mid-number "1,2" and mid-word "back"
        s, e = _snap_word_boundaries(txt, txt.index("1,2") + 2, txt.index("back") + 4)
        self.assertEqual(txt[s:e], "1,212 backlinks")

        # Unclosed parenthesis trimming
        s2, e2 = _trim(txt, txt.index("accident"), len(txt))
        self.assertEqual(txt[s2:e2], "accident-only migration")

    def test_22_existing_external_link_security_audit(self):
        html_doc = '<p>Check <a href="http://insecure-example.com/report">this old report</a> and <a href="https://rival.com/seo">rival guide</a> for details.</p>'
        doc = parse_html(html_doc)
        from lb.site import Page, SiteIndex
        p = Page(key="/test/", source="test.html", title="Test", blocks=doc.blocks, links=doc.links, link_details=doc.link_details)
        temp_idx = SiteIndex([p])
        ext_eng = ExternalLinkEngine(serp_competitor_domains={"rival.com"})
        findings = ext_eng.audit_existing_external_links(temp_idx)
        issues = {f["issue"] for f in findings}
        self.assertIn("INSECURE_HTTP", issues)
        self.assertIn("COMPETITOR_LEAK", issues)

    def test_23_dynamic_canonical_entity_discovery(self):
        mod = check_topical_coherence(
            "Jane Doe",
            "Jane Doe founded the clinical research institute.",
            "/blog/post-1/",
            "/services/consulting/",
            canonical_entities={"jane doe": "/about-jane-doe/"},
        )
        self.assertLessEqual(mod, -50.0)

        mod_ok = check_topical_coherence(
            "Jane Doe",
            "Jane Doe founded the clinical research institute.",
            "/blog/post-1/",
            "/about-jane-doe/",
            canonical_entities={"jane doe": "/about-jane-doe/"},
        )
        self.assertGreater(mod_ok, 0.0)

    def test_24_interactive_5_tab_html_dashboard(self):
        report_html = (self.test_index / "report.html").read_text(encoding="utf-8")
        self.assertIn("1. Internal Link Matrix", report_html)
        self.assertIn("2. External E-E-A-T Citations", report_html)
        self.assertIn("3. Network Graph, PageRank &amp; HITS", report_html)
        self.assertIn("4. Topic Silos &amp; Anchor Entropy", report_html)
        self.assertIn("5. GSC Striking Distance", report_html)


if __name__ == "__main__":
    unittest.main()
