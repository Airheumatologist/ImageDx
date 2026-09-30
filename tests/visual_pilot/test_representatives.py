import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.visual_pilot import db, diseases, representatives, report
from src.visual_pilot.viewer.app import _mark_representatives, create_app


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

    def test_one_per_covered_approved_pair_and_panel_can_cover_many(self):
        panels = [
            self.panel("p1", '["malar_rash", "proposed"]'),
            self.panel("p2", '[{"finding_key":"malar_rash"}]'),
        ]
        chosen = representatives.elect_representatives(panels, self.vocab)
        self.assertEqual([(r["disease_key"], r["finding_key"]) for r in chosen], [("sle", "malar_rash")])
        self.assertEqual(len(chosen), 1)
        self.assertIn("components", json.loads(chosen[0]["scoring_json"]))

    def test_deterministic_panel_id_tie_break(self):
        chosen = representatives.elect_representatives(
            [self.panel("z"), self.panel("a")], self.vocab
        )
        self.assertEqual(chosen[0]["panel_id"], "a")
        reverse = representatives.elect_representatives(
            [self.panel("a"), self.panel("z")], self.vocab
        )
        self.assertEqual(chosen[0]["panel_id"], reverse[0]["panel_id"])
        self.assertEqual(chosen[0]["score"], reverse[0]["score"])

    def test_locked_valid_selection_is_preserved_and_invalid_lock_replaced(self):
        rows = [self.panel("a"), self.panel("b", confidence=0.1)]
        valid = [{"disease_key": "sle", "finding_key": "malar_rash", "panel_id": "b",
                  "selection_source": "manual", "locked": 1}]
        kept = representatives.elect_representatives(rows, self.vocab, valid)[0]
        self.assertEqual((kept["panel_id"], kept["locked"], kept["selection_source"]), ("b", 1, "manual"))
        invalid = [{"disease_key": "sle", "finding_key": "malar_rash", "panel_id": "gone",
                    "selection_source": "manual", "locked": 1}]
        replaced = representatives.elect_representatives(rows, self.vocab, invalid)[0]
        self.assertEqual(replaced["panel_id"], "a")
        self.assertEqual(replaced["selection_source"], "auto_replaced_invalid_lock")
        self.assertEqual(json.loads(replaced["scoring_json"])["replaced_selection"]["panel_id"], "gone")

    def test_score_prefers_typical_confident_clear_high_resolution_whole_image(self):
        good = self.panel("good")
        poor = self.panel("poor", confidence=0.4, typicality="atypical", width=300,
                          height=300, crop_mode="panel", annotations_present=1)
        good_score, audit = representatives.score_panel(good)
        poor_score, _ = representatives.score_panel(poor)
        self.assertGreater(good_score, poor_score)
        self.assertEqual(audit["components"]["crop_integrity"]["points"], 10)

    def test_report_summary_and_api_mapping_serialization(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE manifestation_representatives (disease_key, finding_key, panel_id, score, selection_source, locked)")
        conn.executemany(
            "INSERT INTO manifestation_representatives VALUES (?, ?, ?, ?, ?, ?)",
            [("sle", "malar_rash", "p1", 90, "auto", 0),
             ("sle", "discoid_plaque", "p2", 80, "manual", 1)],
        )
        summary = report.representative_summary(conn)
        self.assertEqual(summary["covered_pairs"], 2)
        self.assertEqual(summary["locked"], 1)
        mapping = representatives.mapping_for_disease(conn, "sle")
        cards = _mark_representatives([{"panel_id": "p1", "panel_ids": ["p1"]}, {"panel_id": "p3"}], mapping)
        self.assertEqual(mapping["malar_rash"]["panel_id"], "p1")
        self.assertEqual(cards[0]["representative_findings"], ["malar_rash"])
        self.assertFalse(cards[1]["is_representative"])

    def test_viewer_startup_rebuilds_mapping_without_report(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            conn = db.connect(data_dir / "visual_pilot.sqlite")
            db.init_db(conn)
            diseases.seed(conn)
            conn.execute("INSERT INTO articles (pmcid, title) VALUES ('PMC_test', 'Cutaneous lupus')")
            conn.execute(
                "INSERT INTO figures (figure_id, pmcid, label, caption, effective_license, status) "
                "VALUES ('PMC_test:fig1', 'PMC_test', 'Figure 1', 'Malar rash in a 23-year-old patient with systemic lupus erythematosus', 'CC-BY', 'stored')"
            )
            conn.execute(
                "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, modality, body_site, "
                "findings_json, typicality, confidence, crop_mode, annotations_present, width, height, "
                "image_path, thumb_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("panel-test", "PMC_test:fig1", "PMC_test", "sle", "clinical_photo", "face",
                 '[{"finding_key":"malar_rash","evidence":"malar rash"}]', "classic", 0.9,
                 "whole_figure", 0, 1200, 900, "panels/test.webp", "thumbs/test.webp"),
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
