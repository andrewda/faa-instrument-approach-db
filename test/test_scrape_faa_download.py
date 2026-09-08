import threading
import time

import pytest

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


def test_get_latest_release_number_iis_style_listing():
    # Mirrors the real FAA directory listing: an IIS-style <pre> block where
    # each entry's timestamp precedes its link and entries are separated by
    # <br> tags. The "CIFP Readme.pdf" entry carries the newest page date.
    html = """
    <html><head><title>aeronav.faa.gov - /Upload_313-d/cifp/</title></head><body>
    <pre><A HREF="/Upload_313-d/">[To Parent Directory]</A><br><br> 8/12/2026 12:43 PM       245900 <A HREF="/Upload_313-d/cifp/CIFP%20Readme.pdf">CIFP Readme.pdf</A><br>  1/2/2025  8:35 AM      8717897 <A HREF="/Upload_313-d/cifp/CIFP_250123.zip">CIFP_250123.zip</A><br> 1/30/2025  9:00 AM      8718931 <A HREF="/Upload_313-d/cifp/CIFP_250220.zip">CIFP_250220.zip</A><br> 8/12/2026  2:46 PM      9109532 <A HREF="/Upload_313-d/cifp/CIFP_260903.zip">CIFP_260903.zip</A><br></pre>
    </body></html>
    """
    session = FakeSession({download.CIFP_URL: html})

    assert download.get_latest_release_number(session=session) == "260903"


def test_get_latest_release_number_ignores_listing_order():
    html = """
    <html><body><pre>
    8/12/2026  2:46 PM      9109532 <A HREF="/Upload_313-d/cifp/CIFP_260903.zip">CIFP_260903.zip</A><br>
    1/2/2025  8:35 AM      8717897 <A HREF="/Upload_313-d/cifp/CIFP_250123.zip">CIFP_250123.zip</A><br>
    7/16/2026  7:30 AM      9098117 <A HREF="/Upload_313-d/cifp/CIFP_260806.zip">CIFP_260806.zip</A><br>
    </pre></body></html>
    """
    session = FakeSession({download.CIFP_URL: html})

    assert download.get_latest_release_number(session=session) == "260903"


def test_get_latest_release_number_no_cifp_zips():
    html = """
    <html><body><pre>
    <A HREF="/Upload_313-d/">[To Parent Directory]</A><br><br>
    8/12/2026 12:43 PM       245900 <A HREF="/Upload_313-d/cifp/CIFP%20Readme.pdf">CIFP Readme.pdf</A><br>
    </body></html>
    """
    session = FakeSession({download.CIFP_URL: html})

    with pytest.raises(ValueError, match="No CIFP zip links found"):
        download.get_latest_release_number(session=session)


def test_get_latest_release_number_ignores_non_conforming_names():
    html = """
    <html><body><pre>
    1/2/2025  8:35 AM      8717897 <A HREF="/Upload_313-d/cifp/CIFP_250123.zip">CIFP_250123.zip</A><br>
    8/12/2026  2:46 PM      9109532 <A HREF="/Upload_313-d/cifp/CIFP_latest.zip">CIFP_latest.zip</A><br>
    8/12/2026  2:46 PM      9109532 <A HREF="/Upload_313-d/cifp/CIFP_250123_backup.zip">CIFP_250123_backup.zip</A><br>
    </pre></body></html>
    """
    session = FakeSession({download.CIFP_URL: html})

    assert download.get_latest_release_number(session=session) == "250123"


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
