import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.visual_pilot import curation, db, diseases
from src.visual_pilot.demographics import resolve_age
from src.visual_pilot.viewer.app import create_app


class AgeEvidenceTests(unittest.TestCase):
    def test_numeric_age_overrides_incorrect_model_label(self):
        age = resolve_age({'caption': 'MRI in a 23-year-old patient with sacroiliitis.'})
        self.assertEqual(age['age_group'], 'adult')
        self.assertEqual(age['patient_age_years'], 23)

    def test_age_boundaries_and_units(self):
        for text, expected in [('11-year-old', 'child'), ('12-year-old', 'adolescent'),
                               ('17-year-old', 'adolescent'), ('18-year-old', 'adult'),
                               ('65-year-old', 'older_adult'), ('18-month-old', 'child')]:
            with self.subTest(text=text):
                self.assertEqual(resolve_age({'caption': f'A {text} patient.'})['age_group'], expected)

    def test_unknown_conflicting_or_unattributed_ages_are_rejected(self):
        for caption in ['MRI of sacroiliitis.', 'Follow-up at 23 years in a patient.',
                        'A 12-year-old patient and a 23-year-old patient.',
                        '(A) A 42-year-old patient. (B) A different patient.',
                        'Juvenile arthritis MRI.']:
            with self.subTest(caption=caption):
                self.assertEqual(resolve_age({'caption': caption})['age_group'], 'unknown')
                self.assertEqual(curation.exclusion_reason(
                    {'include': True, 'age_group': 'adult'}, {'caption': caption}, {}),
                    'patient age unclear or unsupported by source')

    def test_caption_precedes_mentions_and_explicit_age_group_is_supported(self):
        self.assertEqual(resolve_age({'caption': 'An adult patient with lupus.',
                                     'in_text_mentions_json': '["Pediatric patients also develop lupus."]'})['age_group'], 'adult')
        self.assertEqual(resolve_age({'caption': '', 'in_text_mentions_json':
                                     '["The pictured patient is a child."]'})['age_group'], 'child')
        self.assertEqual(resolve_age({'caption': 'MRI showing sacroiliitis.', 'in_text_mentions_json':
                                     '["The pictured patient is a 23-year-old adult."]'})['age_group'], 'adult')

    def test_api_corrects_adult_and_excludes_unknown_across_views(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = db.connect(Path(directory) / db.config.DB_FILENAME)
            db.init_db(conn)
            diseases.seed(conn)
            conn.execute("INSERT INTO articles (pmcid,title) VALUES ('PMCage','Systemic lupus erythematosus')")
            caption = 'Malar rash in a 23-year-old patient with systemic lupus erythematosus. ' + 'Detailed caption sentence. ' * 25 + 'Final caption sentence.'
            for key, age, text in [('adult', 'adolescent', caption),
                                   ('unknown', 'child', 'Malar rash in systemic lupus erythematosus.'),
                                   ('child', 'unknown', 'Malar rash in a 9-year-old patient with systemic lupus erythematosus.')]:
                conn.execute('INSERT INTO figures (figure_id,pmcid,caption,status) VALUES (?,?,?,?)',
                             (key, 'PMCage', text, 'stored'))
                conn.execute('INSERT INTO panels (panel_id,figure_id,pmcid,disease_key,modality,body_site,findings_json,age_group,crop_mode,sha256,width,height) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                             (key,key,'PMCage','sle','clinical_photo','face',json.dumps([{'finding_key':'malar_rash','evidence':'visual'}]),age,'whole_figure',key,900,900))
            conn.commit()
            conn.close()
            with TestClient(create_app(directory)) as client:
                payload = client.get('/api/diseases/sle/panels').json()
                panels = {p['panel_id']: p for p in payload['panels']}
                self.assertEqual(set(panels), {'adult', 'child'})
                self.assertEqual(panels['adult']['age_group'], 'adult')
                self.assertFalse(panels['adult']['pediatric'])
                self.assertTrue(panels['child']['pediatric'])
                self.assertEqual(panels['adult']['source_variants'][0]['figure_caption'], caption)
                self.assertEqual(panels['adult']['context'], caption)
                self.assertLessEqual(len(panels['adult']['context_summary']), 320)
                compared = client.get('/api/compare/sle-dm-skin').json()
                self.assertNotIn('unknown', {p['panel_id'] for entry in compared for side in ('left','right') for p in entry[side].get('panels',[])})


if __name__ == '__main__':
    unittest.main()
