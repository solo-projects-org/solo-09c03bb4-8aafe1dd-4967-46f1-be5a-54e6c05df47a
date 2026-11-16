"""Download-path guarantees: no pinned WAL snapshot, one published run only."""

from __future__ import annotations

import csv
import io

import pytest

from app import db, service


def _wal_size(db_path: str) -> int:
    from pathlib import Path

    path = Path(db_path + "-wal")
    return path.stat().st_size if path.exists() else 0


def _seed_published_run(db_path: str, rows: int = 2500, map_id: int = 1) -> int:
    """Create one published run with ``rows`` result rows (no computation)."""

    db.init_db(db_path)
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            conn.execute(
                """
                INSERT INTO map_versions (filename, sha256, feature_count, created_at, is_active)
                VALUES (?, ?, 1, 'now', 1)
                """,
                (f"map-v{map_id}.geojson", "0" * 64),
            )
            conn.execute(
                """
                INSERT INTO batches (name, filename, sha256, row_count, created_at)
                VALUES ('b', 'hotspots.csv', ?, ?, 'now')
                """,
                ("0" * 64, rows),
            )
            conn.executemany(
                """
                INSERT INTO hotspots
                    (batch_id, seq, source_row, hotspot_no, lon, lat, observed_at_utc, raw_json)
                VALUES (1, ?, ?, ?, 120.1, 30.1, '2026-01-01T00:00:00Z', '{}')
                """,
                [(seq, seq + 2, f"H{seq:05d}") for seq in range(rows)],
            )
            run_id = service._insert_run(conn, 1, map_id)
            conn.execute(
                """
                UPDATE runs SET status = 'succeeded', published = 1, finished_at = 'now',
                    matched_count = ?, total_count = ?
                WHERE id = ?
                """,
                (rows, rows, run_id),
            )
            conn.execute(
                "UPDATE batches SET published_run_id = ? WHERE id = 1", (run_id,)
            )
            conn.executemany(
                """
                INSERT INTO results
                    (run_id, hotspot_seq, status, reason_kind, reason, codes_json)
                VALUES (?, ?, 'matched', 'matched_interior', '唯一落入', '["A-01"]')
                """,
                [(run_id, seq) for seq in range(rows)],
            )
    return run_id


def _parse(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text.lstrip("﻿"))))


def test_paused_export_does_not_pin_the_wal(tmp_path):
    """Consuming only the header must leave SQLite free to checkpoint."""

    db_path = str(tmp_path / "forest.db")
    run_id = _seed_published_run(db_path, rows=2500)

    chunks = service.iter_result_csv(db_path, 1, run_id, chunk=200)
    header = next(chunks)
    assert header.startswith("﻿")
    assert "来源文件" in header and "计算记录号" in header

    # Another instance keeps uploading/writing while the client is paused.
    with db.connection(db_path) as conn:
        for index in range(50):
            with db.write_tx(conn):
                conn.execute(
                    """
                    INSERT INTO batches (name, filename, sha256, row_count, created_at)
                    VALUES (?, 'f', ?, 1, ?)
                    """,
                    (f"concurrent-{index}", "0" * 64, db.utc_now()),
                )
        result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()

    # busy=0 means no reader is holding a snapshot open.
    assert tuple(result)[0] == 0
    assert _wal_size(db_path) == 0

    # The paused export can still be finished and is complete.
    text = header + "".join(chunks)
    rows = _parse(text)
    assert len(rows) == 2501
    assert rows[1][1] == "2"
    assert rows[-1][1] == "2501"


def test_export_keeps_one_run_across_a_mid_download_publication(tmp_path):
    """A recompute published mid-download must not leak into an open export."""

    db_path = str(tmp_path / "forest.db")
    _seed_published_run(db_path, rows=1200)

    chunks = service.iter_result_csv(db_path, 1, 1, chunk=100)
    header = next(chunks)
    text = header + "".join(next(chunks) for _ in range(3))

    # A paused mid-body download also leaves no read transaction open.
    with db.connection(db_path) as conn:
        result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    assert tuple(result)[0] == 0

    # Meanwhile another instance swaps the map and publishes a newer run for
    # the same batch.
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            conn.execute("UPDATE map_versions SET is_active = 0 WHERE is_active = 1")
            conn.execute(
                """
                INSERT INTO map_versions (filename, sha256, feature_count, created_at, is_active)
                VALUES ('map-v2.geojson', ?, 1, 'now', 1)
                """,
                ("1" * 64,),
            )
            new_run = service._insert_run(conn, 1, 2)
            conn.execute(
                """
                UPDATE runs SET status = 'succeeded', published = 1, finished_at = 'now',
                    total_count = 1200 WHERE id = ?
                """,
                (new_run,),
            )
            conn.execute("UPDATE batches SET published_run_id = ? WHERE id = 1", (new_run,))
            conn.executemany(
                """
                INSERT INTO results
                    (run_id, hotspot_seq, status, reason_kind, reason, codes_json)
                VALUES (?, ?, 'unmatched', 'unmatched_outside', '区外', '[]')
                """,
                [(new_run, seq) for seq in range(1200)],
            )

    text += "".join(chunks)
    rows = _parse(text)
    assert len(rows) == 1201
    # Every data row still belongs to run-1 and its v1 map, including the pages
    # read after run-2 was published.
    assert all(row[12] == "run-1" for row in rows[1:])
    assert all(row[11].startswith("v1 ") for row in rows[1:])
    assert "run-2" not in text and "区外" not in text
    # No duplicates and no gaps across the page boundaries.
    assert [row[1] for row in rows[1:]] == [str(seq + 2) for seq in range(1200)]


def test_export_reads_at_most_one_chunk_at_a_time(tmp_path, monkeypatch):
    """Export memory is bounded by the page size, not by the batch size."""

    db_path = str(tmp_path / "forest.db")
    _seed_published_run(db_path, rows=1000)

    page_sizes: list[int] = []
    original = service._read_result_page

    def observed(db_path_, batch_id_, run_id_, after_seq_, limit_):
        rows = original(db_path_, batch_id_, run_id_, after_seq_, limit_)
        page_sizes.append(len(rows))
        return rows

    monkeypatch.setattr(service, "_read_result_page", observed)
    text = "".join(service.iter_result_csv(db_path, 1, 1, chunk=100))
    rows = _parse(text)

    assert len(rows) == 1001
    assert len(page_sizes) == 11  # 10 full pages + one empty final probe
    assert max(page_sizes) <= 100
    assert sum(page_sizes) == 1000


def test_export_rejects_run_from_another_batch(tmp_path):
    db_path = str(tmp_path / "forest.db")
    run_id = _seed_published_run(db_path, rows=5)
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            conn.execute(
                """
                INSERT INTO batches (name, filename, sha256, row_count, created_at)
                VALUES ('other', 'other.csv', ?, 0, 'now')
                """,
                ("0" * 64,),
            )
    with pytest.raises(service.ServiceError):
        list(service.iter_result_csv(db_path, 2, run_id))


def test_export_without_publication_is_rejected(tmp_path):
    db_path = str(tmp_path / "forest.db")
    db.init_db(db_path)
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            conn.execute(
                """
                INSERT INTO batches (name, filename, sha256, row_count, created_at)
                VALUES ('b', 'f', ?, 0, 'now')
                """,
                ("0" * 64,),
            )
    with pytest.raises(service.ServiceError):
        list(service.iter_result_csv(db_path, 1))
