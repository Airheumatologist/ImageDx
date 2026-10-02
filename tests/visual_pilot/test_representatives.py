import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.visual_pilot import diseases, representatives, report
from src.visual_pilot.viewer.app import _mark_representatives, create_app
from balanced_fixtures import (
    MALAR_CAPTION,
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_panel,
    make_db,
)

DISCOID_CAPTION = (
    "Discoid plaque in a 40-year-old patient with systemic lupus erythematosus"
)


class RepresentativeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.vocab = [
            {"finding_key": "malar_rash", "disease_keys_json": '["sle"]', "approved": 1},
            {"finding_key": "proposed", "disease_keys_json": '["sle"]', "approved": 0},
            {"finding_key": "heliotrope_rash", "disease_keys_json": '["dm"]', "approved": 1},
        ]

    def panel(self, panel_id, findings='["malar_rash"]', **extra):
        return {
            "panel_id": panel_id, "disease_key": "sle", "findings_json": findings,
            "confidence": 0.8, "typicality": "classic", "width": 900, "height": 900,
            "crop_mode": "whole_figure", "annotations_present": 0, **extra,
        }

    def test_score_prefers_typical_confident_clear_high_resolution_whole_image(self):
        good = self.panel("good")
        poor = self.panel("poor", confidence=0.4, typicality="atypical", width=300,
                          height=300, crop_mode="panel", annotations_present=1)
        good_score, audit = representatives.score_panel(good)
        poor_score, _ = representatives.score_panel(poor)
        self.assertGreater(good_score, poor_score)
        self.assertEqual(audit["components"]["crop_integrity"]["points"], 10)

    def _seed_pair(self, conn, data_dir, panel_id="p1", figure_id="PMC1:fig1",
                   findings=("malar_rash",), caption=None, sha256="sha-1",
                   confidence=0.9):
        add_disease(conn, "sle", "Systemic lupus erythematosus")
        add_article(conn, "PMC1")
        add_figure(conn, figure_id, "PMC1", caption=caption or MALAR_CAPTION)
        add_panel(
            conn, data_dir, panel_id, figure_id, "PMC1", "sle",
            findings=findings, sha256=sha256, confidence=confidence,
        )

    def test_report_summary_and_api_mapping_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = make_db(data_dir)
            add_disease(conn, "sle", "Systemic lupus erythematosus")
            add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
            add_finding(conn, "discoid_plaque", ("sle",), label="Discoid plaque")
            self._seed_pair(conn, data_dir, "p1")
            self._seed_pair(
                conn, data_dir, "p2", figure_id="PMC1:fig2",
                findings=("discoid_plaque",), caption=DISCOID_CAPTION,
                sha256="sha-2",
            )
            conn.execute(
                "INSERT INTO manifestation_representatives"
                "(disease_key,finding_key,panel_id,score,selection_source,locked) "
                "VALUES('sle','discoid_plaque','p2',80,'manual',1)"
            )
            conn.commit()
            representatives.rebuild(conn)
            summary = report.representative_summary(conn)
            self.assertEqual(summary["covered_pairs"], 2)
            self.assertEqual(summary["locked"], 1)
            mapping = representatives.mapping_for_disease(conn, "sle")
            cards = _mark_representatives(
                [{"panel_id": "p1", "panel_ids": ["p1"]}, {"panel_id": "p3"}], mapping
            )
            self.assertEqual(mapping["malar_rash"]["panel_id"], "p1")
            self.assertEqual(mapping["discoid_plaque"]["panel_id"], "p2")
            self.assertTrue(mapping["discoid_plaque"]["locked"])
            self.assertEqual(cards[0]["representative_findings"], ["malar_rash"])
            self.assertFalse(cards[1]["is_representative"])
            conn.close()

    def test_rebuild_replaces_invalid_lock_and_preserves_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = make_db(data_dir)
            add_disease(conn, "sle", "Systemic lupus erythematosus")
            add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
            self._seed_pair(conn, data_dir, "p1")
            conn.execute(
                "INSERT INTO manifestation_representatives"
                "(disease_key,finding_key,panel_id,score,scoring_json,"
                "selection_source,locked,updated_at) "
                "VALUES('sle','malar_rash','gone',55,'{\"stale\": true}',"
                "'manual',1,'2026-09-01 00:00:00')"
            )
            conn.commit()
            result = representatives.rebuild(conn)
            self.assertEqual(result["auto_replaced"], 1)
            row = conn.execute(
                "SELECT panel_id,selection_source,locked,scoring_json "
                "FROM manifestation_representatives"
            ).fetchone()
            self.assertEqual((row["panel_id"], row["selection_source"], row["locked"]),
                             ("p1", "auto_replaced_invalid_lock", 0))
            replaced = json.loads(row["scoring_json"])["replaced_selection"]
            # The full original locked payload survives the replacement.
            self.assertEqual(replaced["panel_id"], "gone")
            self.assertEqual(replaced["selection_source"], "manual")
            self.assertEqual(replaced["locked"], 1)
            self.assertEqual(replaced["scoring_json"], '{"stale": true}')
            conn.close()

    def test_locked_row_survives_pair_losing_all_images_but_maps_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = make_db(data_dir)
            add_disease(conn, "sle", "Systemic lupus erythematosus")
            add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
            self._seed_pair(conn, data_dir, "p1")
            conn.execute(
                "INSERT INTO manifestation_representatives"
                "(disease_key,finding_key,panel_id,score,selection_source,locked) "
                "VALUES('sle','malar_rash','p1',90,'manual',1)"
            )
            conn.execute(
                "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,"
                "policy_version) VALUES('p1','sha-1','exclude','audit','v1')"
            )
            conn.commit()
            representatives.rebuild(conn)
            row = conn.execute(
                "SELECT panel_id,locked,selection_source "
                "FROM manifestation_representatives WHERE finding_key='malar_rash'"
            ).fetchone()
            # Inactive record retained; it is not a publishable primary.
            self.assertEqual(tuple(row), ("p1", 1, "manual"))
            self.assertNotIn("malar_rash", representatives.mapping_for_disease(conn, "sle"))
            conn.close()

    def test_lock_history_survives_rebuilds_and_curation_returns_new_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = make_db(data_dir)
            add_disease(conn, "sle", "Systemic lupus erythematosus")
            add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
            self._seed_pair(conn, data_dir, "p_low", figure_id="PMC1:fig1",
                            sha256="sha-low", confidence=0.2)
            self._seed_pair(conn, data_dir, "p_high", figure_id="PMC1:fig2",
                            sha256="sha-high", confidence=0.95)
            conn.execute(
                "INSERT INTO manifestation_representatives"
                "(disease_key,finding_key,panel_id,score,selection_source,locked) "
                "VALUES('sle','malar_rash','p_low',70,'manual',1)"
            )
            conn.commit()
            representatives.rebuild(conn)
            # Valid lock leads while its panel is published.
            self.assertEqual(
                representatives.mapping_for_disease(conn, "sle")
                ["malar_rash"]["panel_id"], "p_low")
            # Curation excludes the locked panel — the surviving candidate
            # becomes the primary without any startup/report intervention.
            conn.execute(
                "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,"
                "policy_version) VALUES('p_low','sha-low','exclude','audit','v1')"
            )
            conn.commit()
            representatives.rebuild(conn)
            row = conn.execute(
                "SELECT panel_id,selection_source,locked,scoring_json "
                "FROM manifestation_representatives WHERE finding_key='malar_rash'"
            ).fetchone()
            self.assertEqual(
                (row["panel_id"], row["selection_source"], row["locked"]),
                ("p_high", "auto_replaced_invalid_lock", 0))
            replaced = json.loads(row["scoring_json"])["replaced_selection"]
            self.assertEqual(replaced["panel_id"], "p_low")
            self.assertEqual(replaced["selection_source"], "manual")
            # A second rebuild propagates the replacement history unchanged.
            representatives.rebuild(conn)
            row2 = conn.execute(
                "SELECT panel_id,scoring_json FROM manifestation_representatives "
                "WHERE finding_key='malar_rash'"
            ).fetchone()
            self.assertEqual(row2["panel_id"], "p_high")
            self.assertEqual(
                json.loads(row2["scoring_json"])["replaced_selection"], replaced)
            mapping = representatives.mapping_for_disease(conn, "sle")
            self.assertEqual(mapping["malar_rash"]["panel_id"], "p_high")
            conn.close()

    def test_rebuild_is_deterministic_and_uses_gallery_lead(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = make_db(data_dir)
            add_disease(conn, "sle", "Systemic lupus erythematosus")
            add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
            # Lower-scored panel ids sort later; the score leader must win.
            self._seed_pair(conn, data_dir, "z_low", figure_id="PMC1:fig1",
                            sha256="sha-z", confidence=0.2)
            self._seed_pair(conn, data_dir, "a_high", figure_id="PMC1:fig2",
                            sha256="sha-a", confidence=0.95)
            conn.commit()
            first = representatives.rebuild(conn)
            rows1 = [
                tuple(r)
                for r in conn.execute(
                    "SELECT * FROM manifestation_representatives ORDER BY finding_key"
                )
            ]
            second = representatives.rebuild(conn)
            rows2 = [
                tuple(r)
                for r in conn.execute(
                    "SELECT * FROM manifestation_representatives ORDER BY finding_key"
                )
            ]
            self.assertEqual(first, second)
            self.assertEqual([r[2] for r in rows1], ["a_high"])
            self.assertEqual(rows1, rows2)
            mapping = representatives.mapping_for_disease(conn, "sle")
            self.assertEqual(mapping["malar_rash"]["panel_id"], "a_high")
            conn.close()

    def test_viewer_startup_rebuilds_mapping_without_report(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = make_db(data_dir)
            diseases.seed(conn)
            add_article(conn, "PMC_test", title="Cutaneous lupus")
            add_figure(conn, "PMC_test:fig1", "PMC_test")
            add_panel(
                conn, data_dir, "panel-test", "PMC_test:fig1", "PMC_test", "sle",
                findings=("malar_rash",), sha256="sha-viewer",
            )
            conn.commit()
            conn.close()

            app = create_app(str(data_dir))
            with TestClient(app) as client:
                response = client.get("/api/diseases/sle/panels")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["representatives"]["malar_rash"]["panel_id"], "panel-test")
            self.assertTrue(any(p["is_representative"] for p in payload["panels"]))


if __name__ == "__main__":
    unittest.main()
