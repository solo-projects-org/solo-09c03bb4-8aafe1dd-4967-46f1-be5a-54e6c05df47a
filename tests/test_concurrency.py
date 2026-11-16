"""Cross-process concurrency tests: real Uvicorn servers, one database.

These exercise the guarantee that matters in deployment: two web instances
uploading/recomputing at the same time must never publish a detail file that
mixes results from two different compartment versions.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    def __init__(self, db_path: Path, port: int, cwd: Path):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env.update(
            FOREST_DB_PATH=str(db_path),
            FOREST_RUN_WORKER="1",
            FOREST_STALE_RUN_SECONDS="2",
            PYTHONPATH=str(ROOT),
        )
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 25.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(self.base + "/healthz", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.proc.kill()


def upload_map(client: httpx.Client, name: str, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/maps",
        files={"file": (name, payload, "application/geo+json")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def upload_hotspots(client: httpx.Client, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/batches",
        files={"file": ("hotspots.csv", payload, "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def wait_published(client: httpx.Client, batch_id: int, timeout: float = 25.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"{client.base_url}/api/batches/{batch_id}").json()
        if payload["published_total"] is not None and not payload["pending"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("batch never published")


@pytest.mark.slow
def test_two_servers_publish_single_consistent_version(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            first = wait_published(client_a, 1)
            assert first["published_map_version"] == 1

            # Instance A swaps the map while instance B triggers a recompute.
            upload_map(client_a, "v2.geojson", (SAMPLES / "compartments_v2.geojson").read_bytes())
            response = client_b.post(
                f"{client_b.base_url}/batches/1/recompute", follow_redirects=False
            )
            assert response.status_code == 303

            final = wait_published(client_b, 1)
            assert final["published_map_version"] == 2
            assert final["published_total"] == 15

            # The published CSV must be internally consistent: it is produced
            # exclusively from the run pinned to the active v2 map.
            csv_bytes = client_a.get(f"{client_a.base_url}/batches/1/download.csv").content
            text = csv_bytes.decode("utf-8-sig")
            rows = text.splitlines()
            assert len(rows) == 16
            assert all("v2" in line for line in rows[1:])
            assert "v1" not in text
            assert "A-01;B-02" not in text  # v1-only shared edge between A and B

            # Every hotspot row present exactly once.
            assert text.count("H-00") == 15

            # No computation against the retired v1 map may remain pending, and
            # no v1 run may be the published one.
            payload = client_a.get(f"{client_a.base_url}/api/batches/1").json()
            active_version = client_a.get(f"{client_a.base_url}/healthz").json()[
                "active_map_version"
            ]
            assert payload["published_map_version"] == active_version == 2
            for run in payload["runs"]:
                if run["map_version_id"] == 1:
                    assert run["status"] in {"succeeded", "superseded"}
            published_runs = [
                run for run in payload["runs"] if run["id"] == payload["published_run_id"]
            ]
            assert published_runs[0]["map_version_id"] == active_version
    finally:
        server_a.stop()
        server_b.stop()


@pytest.mark.slow
def test_slow_download_does_not_block_wal_checkpoint_or_writes(tmp_path):
    """A client paused at the header must not pin the WAL snapshot.

    Instance A streams a large CSV; the client reads only the status line and
    headers (``http.client`` does not read ahead) and then stalls.  Instance B
    must still be able to upload hotspots and publish a computation, and a WAL
    checkpoint must succeed and truncate the log while the client is stalled.
    """

    import http.client
    import sqlite3

    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        hotspot_rows = 80000
        buf = ["hotspot_no,lon,lat,observed_at"]
        buf += [f"H{i:05d},0.5,0.5,2026-01-01T00:00:00Z" for i in range(hotspot_rows)]
        hotspot_csv = ("\n".join(buf) + "\n").encode()

        with httpx.Client(base_url=server_a.base, timeout=30) as writer:
            upload_map(writer, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(writer, hotspot_csv)
            wait_published(writer, 1, timeout=60)

        # Pause a download after the response headers only.
        paused = http.client.HTTPConnection("127.0.0.1", port_a, timeout=60)
        paused.request("GET", "/batches/1/download.csv")
        response = paused.getresponse()
        assert response.status == 200
        assert response.getheader("content-type").startswith("text/csv")
        assert response.getheader("content-disposition").startswith("attachment")
        assert response.read(0) == b""  # make sure http.client did not consume the body

        with httpx.Client(base_url=server_b.base, timeout=60) as other:
            # Writes and publication continue while the download is paused.
            upload_map(other, "v2.geojson", (SAMPLES / "compartments_v2.geojson").read_bytes())
            response = other.post(f"{other.base_url}/batches/1/recompute", follow_redirects=False)
            assert response.status_code == 303
            deadline = time.time() + 60
            while time.time() < deadline:
                payload = other.get(f"{other.base_url}/api/batches/1").json()
                if payload["published_map_version"] == 2 and not payload["pending"]:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("recompute did not publish while a download was paused")
            assert payload["published_total"] == hotspot_rows

        wal_path = tmp_path / "forest.db-wal"
        wal_before = wal_path.stat().st_size if wal_path.exists() else 0
        assert wal_before < 16 * 1024 * 1024, wal_before

        # No open read snapshot: the WAL can be checkpointed and truncated.
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            busy, log_frames, checkpointed = conn.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
        assert busy == 0, (busy, log_frames, checkpointed)
        assert (wal_path.stat().st_size if wal_path.exists() else 0) == 0

        # Abruptly drop the paused download; the server must release it and
        # keep serving further requests normally.
        paused.close()

        with httpx.Client(base_url=server_a.base, timeout=30) as client:
            deadline = time.time() + 30
            while time.time() < deadline:
                download = client.get(f"{client.base_url}/batches/1/download.csv")
                if download.status_code == 200:
                    break
                time.sleep(0.2)
            assert download.status_code == 200
            text = download.content.decode("utf-8-sig")
            lines = text.splitlines()
            assert len(lines) == hotspot_rows + 1
            assert "run-2" in lines[1]
            assert "run-1" not in text

        with sqlite3.connect(str(db_path), timeout=30) as conn:
            busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert busy == 0
    finally:
        server_a.stop()
        server_b.stop()


@pytest.mark.slow
def test_concurrent_map_swaps_leave_exactly_one_active(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_hotspots(client_a, (SAMPLES / "hotspots.csv").read_bytes())
            v1 = (SAMPLES / "compartments_v1.geojson").read_bytes()
            v2 = (SAMPLES / "compartments_v2.geojson").read_bytes()

            import threading

            errors: list[BaseException] = []

            def swap(client, payload, name):
                try:
                    upload_map(client, name, payload)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=swap, args=(client_a, v1, "a.geojson")),
                threading.Thread(target=swap, args=(client_b, v2, "b.geojson")),
                threading.Thread(target=swap, args=(client_a, v2, "c.geojson")),
                threading.Thread(target=swap, args=(client_b, v1, "d.geojson")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert not errors

            health = client_a.get(f"{client_a.base_url}/healthz").json()
            assert health["active_map_version"] is not None
            from app import db, service

            with db.connection(str(db_path)) as conn:
                active_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM map_versions WHERE is_active = 1"
                ).fetchone()["n"]
                versions = conn.execute("SELECT COUNT(*) AS n FROM map_versions").fetchone()["n"]
            assert active_count == 1
            assert versions == 4

            # The active version is the one the health endpoint reports, and a
            # recompute publishes against it with no mixed output.
            client_b.post(f"{client_b.base_url}/batches/1/recompute", follow_redirects=False)
            final = wait_published(client_b, 1)
            assert final["published_map_version"] == health["active_map_version"]
    finally:
        server_a.stop()
        server_b.stop()
