"""Preserve approved pilot panels and text evidence in a fresh data run.

The new database remains authoritative except where a previously approved
panel disappeared or lost its supported finding during regeneration. This
command makes a SQLite backup before changing the fresh database and keeps
restored panel media under separate ``legacy`` paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config, db
from .curation_audit import audit
from .extract_findings import rebuild_image_rows


def _upsert(conn, table: str, row: dict, key: str) -> None:
    columns = list(row)
    values = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{col}=excluded.{col}" for col in columns if col != key)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values}) "
        f"ON CONFLICT({key}) DO UPDATE SET {updates}",
        [row[col] for col in columns],
    )


def _copy_panel_file(old_root: Path, new_root: Path, relative: str | None,
                     disease: str) -> str | None:
    if not relative:
        return relative
    source_path = Path(relative)
    if source_path.is_absolute() or ".." in source_path.parts or source_path.parts[0] not in {"panels", "thumbs"}:
        raise ValueError(f"Unsafe panel media path: {relative}")
    source = old_root / source_path
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_rel = Path(source_path.parts[0]) / "legacy" / disease / source_path.name
    destination = new_root / destination_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Legacy media collision: {destination}")
    else:
        shutil.copy2(source, destination)
    return destination_rel.as_posix()


def merge(old_root: Path, new_root: Path) -> dict:
    old_root, new_root = old_root.resolve(), new_root.resolve()
    if old_root == new_root:
        raise ValueError("Old and new data directories must differ")
    old_db = old_root / config.DB_FILENAME
    new_db = new_root / config.DB_FILENAME
    if not old_db.is_file() or not new_db.is_file():
        raise FileNotFoundError("Both data directories need an existing database")

    old = db.connect(old_db)
    new = db.connect(new_db)
    try:
        old_approved = {
            item["panel_id"] for item in audit(old)["decisions"]
            if item["decision"] == "retain"
        }
        new_approved = {
            item["panel_id"] for item in audit(new)["decisions"]
            if item["decision"] == "retain"
        }
        restore = sorted(old_approved - new_approved)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = new_root / "reports" / f"pre_merge_{stamp}.sqlite"
        backup.parent.mkdir(parents=True, exist_ok=True)
        with db.connect(backup) as copy:
            new.backup(copy)

        # Copy files before the transaction so every inserted media path exists.
        rows = []
        for panel_id in restore:
            panel = dict(old.execute("SELECT * FROM panels WHERE panel_id=?", (panel_id,)).fetchone())
            panel["image_path"] = _copy_panel_file(
                old_root, new_root, panel["image_path"], panel["disease_key"]
            )
            panel["thumb_path"] = _copy_panel_file(
                old_root, new_root, panel["thumb_path"], panel["disease_key"]
            )
            rows.append(panel)

        with new:
            for panel in rows:
                article = dict(old.execute("SELECT * FROM articles WHERE pmcid=?", (panel["pmcid"],)).fetchone())
                figure = dict(old.execute("SELECT * FROM figures WHERE figure_id=?", (panel["figure_id"],)).fetchone())
                new.execute(
                    f"INSERT OR IGNORE INTO articles ({', '.join(article)}) VALUES ({', '.join('?' for _ in article)})",
                    list(article.values()),
                )
                current_figure = new.execute(
                    "SELECT status FROM figures WHERE figure_id=?", (panel["figure_id"],)
                ).fetchone()
                if current_figure is None or current_figure["status"] != "stored":
                    _upsert(new, "figures", figure, "figure_id")
                _upsert(new, "panels", panel, "panel_id")

            # Keep distinct previously extracted article statements. New
            # approved vocabulary wins; old proposed keys are added only when
            # missing so citations stay interpretable.
            for row in old.execute("SELECT * FROM findings_vocab"):
                record = dict(row)
                new.execute(
                    f"INSERT OR IGNORE INTO findings_vocab ({', '.join(record)}) "
                    f"VALUES ({', '.join('?' for _ in record)})", list(record.values()),
                )
            finding_cols = (
                "disease_key", "finding_key", "subtype", "frequency_text",
                "frequency_pct_low", "frequency_pct_high", "source", "pmcid", "quote",
            )
            existing = {
                tuple(row) for row in new.execute(
                    f"SELECT {', '.join(finding_cols)} FROM disease_findings WHERE source='text'"
                )
            }
            text_added = 0
            for row in old.execute(
                f"SELECT {', '.join(finding_cols)} FROM disease_findings WHERE source='text'"
            ):
                value = tuple(row)
                if value in existing:
                    continue
                new.execute(
                    f"INSERT INTO disease_findings ({', '.join(finding_cols)}) "
                    f"VALUES ({', '.join('?' for _ in finding_cols)})", value,
                )
                existing.add(value)
                text_added += 1
            image_rows = rebuild_image_rows(new)

        return {
            "restored_panels": len(rows),
            "restored_panel_ids": restore,
            "text_findings_added": text_added,
            "image_findings": image_rows,
            "backup_path": str(backup),
        }
    finally:
        old.close()
        new.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_data_dir", type=Path)
    parser.add_argument("new_data_dir", type=Path)
    args = parser.parse_args()
    result = merge(args.old_data_dir, args.new_data_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
