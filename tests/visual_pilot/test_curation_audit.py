"""Exclusions remain reversible and do not apply to replacement images."""

from src.visual_pilot import curation_audit, db


def _seed(conn):
    conn.execute("INSERT INTO articles(pmcid,title) VALUES ('PMC1','Lupus images')")
    conn.execute("INSERT INTO figures(figure_id,pmcid,caption) VALUES ('PMC1:f1','PMC1','Malar rash')")
    for pid, sha in [('keep', 'keep-sha'), ('exclude', 'exclude-sha')]:
        conn.execute(
            'INSERT INTO panels(panel_id,figure_id,pmcid,disease_key,sha256) VALUES (?,?,?,?,?)',
            (pid, 'PMC1:f1', 'PMC1', None, sha),
        )
    conn.commit()


def test_audit_apply_is_idempotent_and_preserves_source(conn, monkeypatch):
    _seed(conn)
    monkeypatch.setattr(curation_audit.curation, 'exclusion_reason',
                        lambda p, f, a: 'text-only chart' if p['panel_id'] == 'exclude' else None)
    report = curation_audit.audit(conn)
    assert report['total'] == 2 and report['excluded'] == 1
    assert conn.execute('SELECT COUNT(*) FROM panel_curation').fetchone()[0] == 0
    assert curation_audit.apply_audit(conn, report) == 1
    assert curation_audit.apply_audit(conn, report) == 1
    assert conn.execute('SELECT COUNT(*) FROM panel_curation').fetchone()[0] == 1
    assert conn.execute('SELECT COUNT(*) FROM panels').fetchone()[0] == 2
    assert [r['panel_id'] for r in conn.execute('SELECT * FROM published_panels')] == ['keep']
    # Audit exclusion applies only to exactly the image reviewed.
    conn.execute("UPDATE panels SET sha256 = 'new-sha' WHERE panel_id='exclude'")
    assert conn.execute('SELECT COUNT(*) FROM published_panels').fetchone()[0] == 2
    # An audit from before the replacement cannot hide its new image.
    assert curation_audit.apply_audit(conn, report) == 0


def test_policy_reversal_restores_publication(conn, monkeypatch):
    _seed(conn)
    monkeypatch.setattr(curation_audit.curation, 'exclusion_reason', lambda p, f, a: 'collage')
    curation_audit.apply_audit(conn, curation_audit.audit(conn))
    assert conn.execute('SELECT COUNT(*) FROM published_panels').fetchone()[0] == 0
    monkeypatch.setattr(curation_audit.curation, 'exclusion_reason', lambda p, f, a: None)
    curation_audit.apply_audit(conn, curation_audit.audit(conn))
    assert conn.execute('SELECT COUNT(*) FROM published_panels').fetchone()[0] == 2


def test_schema_upgrade_preserves_legacy_panel_rows(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    conn = db.init_db(db.connect(path))
    _seed(conn)
    conn.execute('DROP VIEW published_panels')
    conn.execute('DROP TABLE panel_curation')
    conn.commit()
    db.init_db(conn)
    assert conn.execute('SELECT COUNT(*) FROM published_panels').fetchone()[0] == 2
    conn.close()


def test_excluded_images_do_not_satisfy_coverage_or_generate_findings(conn, monkeypatch):
    from src.visual_pilot import cli, extract_findings, report, select_articles

    _seed(conn)
    conn.execute("INSERT INTO diseases(disease_key,name) VALUES ('sle','Lupus')")
    conn.execute(
        "INSERT INTO findings_vocab(finding_key,label,category,approved,disease_keys_json) "
        "VALUES ('malar_rash','Malar rash','skin',1,'[\"sle\"]')"
    )
    conn.execute(
        "UPDATE panels SET disease_key='sle', findings_json='[{\"finding_key\":\"malar_rash\"}]' "
        "WHERE panel_id='exclude'"
    )
    monkeypatch.setattr(curation_audit.curation, 'exclusion_reason',
                        lambda p, f, a: 'collage' if p['panel_id'] == 'exclude' else None)
    curation_audit.apply_audit(conn, curation_audit.audit(conn))
    assert select_articles._stored_panel_counts(conn, 'sle') == {}
    assert cli._snapshot(conn, 'sle') == (set(), set())
    assert extract_findings.rebuild_image_rows(conn) == 0
    assert report.funnel(conn, 'sle')['excluded_panels'] == 1
    assert report.zero_image_findings(conn)['sle'] == ['malar_rash']
