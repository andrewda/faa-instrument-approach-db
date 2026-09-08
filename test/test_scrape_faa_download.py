from datetime import datetime
import threading
import time

from scrape_faa import download


class FakeResponse:
    def __init__(self, *, text="", headers=None, chunks=None):
        self.text = text
        self.headers = headers or {}
        self._chunks = chunks or [b"data"]

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        return iter(self._chunks)


class FakeSession:
    def __init__(self, html_by_url):
        self.html_by_url = html_by_url

    def get(self, url, timeout=None, stream=False):
        del timeout
        del stream
        return FakeResponse(text=self.html_by_url[url])

    def close(self):
        return None


def test_get_latest_release_number_uses_page_timestamps(monkeypatch):
    html = """
    <html><body><pre>
    <a href="CIFP_250101.zip">CIFP_250101.zip</a> 2025-01-01 10:00
    <a href="CIFP_250102.zip">CIFP_250102.zip</a> 2025-01-02 09:00
    </pre></body></html>
    """
    session = FakeSession({download.CIFP_URL: html})

    def fail_head_request(url):
        raise AssertionError(f"unexpected HEAD fallback for {url}")

    monkeypatch.setattr(download, "get_file_timestamp", fail_head_request)

    assert download.get_latest_release_number(session=session) == "250102"


def test_get_latest_release_number_falls_back_to_head(monkeypatch):
    html = """
    <html><body><pre>
    <a href="CIFP_250101.zip">CIFP_250101.zip</a>
    <a href="CIFP_250102.zip">CIFP_250102.zip</a>
    </pre></body></html>
    """
    session = FakeSession({download.CIFP_URL: html})

    requested = []

    def fake_head_timestamp(url):
        requested.append(url)
        if "250101" in url:
            return datetime(2025, 1, 1, 10, 0)
        return datetime(2025, 1, 2, 9, 0)

    monkeypatch.setattr(download, "get_file_timestamp", fake_head_timestamp)

    assert download.get_latest_release_number(session=session) == "250102"
    assert len(requested) == 2


def test_download_dtpp_zips_parallel_workers(monkeypatch, tmp_path):
    release = "250123"
    html = f"""
    <html><body>
    <a href="DDTPPA_{release}.zip">A</a>
    <a href="DDTPPB_{release}.zip">B</a>
    <a href="DDTPPC_{release}.zip">C</a>
    <a href="DDTPPD_{release}.zip">D</a>
    </body></html>
    """
    session = FakeSession({download.DTPP_URL: html})

    lock = threading.Lock()
    active_workers = 0
    max_active_workers = 0

    def fake_download_file(zip_url, download_folder, session=None):
        del zip_url
        del download_folder
        del session
        nonlocal active_workers
        nonlocal max_active_workers
        with lock:
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
        time.sleep(0.05)
        with lock:
            active_workers -= 1
        return "ok"

    monkeypatch.setattr(download, "_download_file", fake_download_file)

    download.download_dtpp_zips(
        release,
        str(tmp_path),
        max_workers=4,
        session=session,
    )
    assert max_active_workers > 1


def test_download_dtpp_zips_single_worker(monkeypatch, tmp_path):
    release = "250123"
    html = f"""
    <html><body>
    <a href="DDTPPA_{release}.zip">A</a>
    <a href="DDTPPB_{release}.zip">B</a>
    </body></html>
    """
    session = FakeSession({download.DTPP_URL: html})

    lock = threading.Lock()
    active_workers = 0
    max_active_workers = 0

    def fake_download_file(zip_url, download_folder, session=None):
        del zip_url
        del download_folder
        del session
        nonlocal active_workers
        nonlocal max_active_workers
        with lock:
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
        time.sleep(0.01)
        with lock:
            active_workers -= 1
        return "ok"

    monkeypatch.setattr(download, "_download_file", fake_download_file)

    download.download_dtpp_zips(
        release,
        str(tmp_path),
        max_workers=1,
        session=session,
    )
    assert max_active_workers == 1
