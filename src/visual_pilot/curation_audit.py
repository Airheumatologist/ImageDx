"""Audit already stored images using the publication policy, without deletion.

Run ``python3 -m src.visual_pilot.curation_audit`` for a read-only report;
add ``--apply`` to save reversible, image-hash-bound publication exclusions.
No model calls, network access, image edits or changes to source judgments.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config, curation, db, diseases


def audit(conn, disease: str | None = None) -> dict:
    articles = {r['pmcid']: dict(r) for r in conn.execute('SELECT * FROM articles')}
    figures = {r['figure_id']: dict(r) for r in conn.execute('SELECT * FROM figures')}
    rows = conn.execute(
        'SELECT * FROM panels' + (' WHERE disease_key = ?' if disease else ''),
        (disease,) if disease else (),
    )
    decisions = []
    counts = Counter()
    for row in rows:
        panel = dict(row)
        reason = curation.exclusion_reason(
            panel, figures.get(panel['figure_id'], {}), articles.get(panel['pmcid'], {}),
        )
        counts[panel['disease_key']] += 1
        decisions.append({
            'panel_id': panel['panel_id'],
            'figure_id': panel['figure_id'],
            'disease_key': panel['disease_key'],
            'image_sha256': panel['sha256'] or '',
            'decision': 'exclude' if reason else 'retain',
            'reason': reason,
        })
    excluded = [d for d in decisions if d['decision'] == 'exclude']
    return {
        'policy_version': curation.POLICY_VERSION,
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'total': len(decisions),
        'excluded': len(excluded),
        'retained': len(decisions) - len(excluded),
        'by_disease': {
            k: {'total': n, 'excluded': sum(d['disease_key'] == k for d in excluded)}
            for k, n in sorted(counts.items(), key=lambda item: str(item[0] or ''))
        },
        'reasons': dict(Counter(d['reason'] for d in excluded)),
        'decisions': decisions,
    }


def apply_audit(conn, report: dict) -> int:
    """Replace automated reviews for scanned images; stale reviews cannot hide
    a replacement image. Source rows and files are retained for inspection.
    """
    saved = 0
    with conn:
        for item in report['decisions']:
            current = conn.execute(
                'SELECT sha256 FROM panels WHERE panel_id = ?', (item['panel_id'],)
            ).fetchone()
            if current is None or (current['sha256'] or '') != item['image_sha256']:
                continue
            if item['decision'] == 'exclude':
                conn.execute(
                    'INSERT INTO panel_curation '
                    '(panel_id, image_sha256, decision, reason, policy_version) '
                    'VALUES (?, ?, ?, ?, ?) ON CONFLICT(panel_id) DO UPDATE SET '
                    'image_sha256=excluded.image_sha256, decision=excluded.decision, '
                    'reason=excluded.reason, policy_version=excluded.policy_version, '
                    "reviewed_at=datetime('now')",
                    (item['panel_id'], item['image_sha256'], 'exclude', item['reason'],
                     report['policy_version']),
                )
                saved += 1
            else:
                conn.execute('DELETE FROM panel_curation WHERE panel_id = ?', (item['panel_id'],))
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=config.data_dir())
    parser.add_argument('--disease', choices=[*diseases.load_diseases(), 'all'], default='all')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    database = root / config.DB_FILENAME
    # A dry run never creates tables or writes to the source database.
    if not args.apply:
        conn = sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
    else:
        conn = db.connect(database)
    try:
        report = audit(conn, None if args.disease == 'all' else args.disease)
        if args.apply:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup_path = root / 'reports' / f'pre_curation_{stamp}.sqlite'
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(backup_path) as backup:
                conn.backup(backup)
            db.init_db(conn)
            report['saved_exclusions'] = apply_audit(conn, report)
            from .extract_findings import rebuild_image_rows

            report['published_image_findings'] = rebuild_image_rows(conn)
            conn.commit()
            report['backup_path'] = str(backup_path)
        target = root / 'reports' / f'curation_audit_{args.disease}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'decisions'}, indent=2))
        print(f'Audit report: {target}')
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
