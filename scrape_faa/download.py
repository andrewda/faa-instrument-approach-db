import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import argparse
import re


CIFP_URL = "https://aeronav.faa.gov/Upload_313-d/cifp/"
DTPP_URL = "https://aeronav.faa.gov/upload_313-d/terminal/"
CIFP_ZIP_NAME_PATTERN = re.compile(r"^CIFP_(\d{6})\.zip$")
DEFAULT_TIMEOUT = 10
DEFAULT_DOWNLOAD_WORKERS = 8
CHUNK_SIZE_BYTES = 1024 * 20


def _create_session() -> requests.Session:
    return requests.Session()


def _fetch_zip_links(base_url: str, html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    zip_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.endswith(".zip"):
            continue
        file_name = os.path.basename(href)
        zip_links.append((file_name, urljoin(base_url, href)))
    return zip_links


def get_latest_release_number(session: Optional[requests.Session] = None) -> str:
    # Uses the CIFP url to get the latest d-TPP release number like
    # 250320, 250417, etc.
    print(f"Fetching page: {CIFP_URL}")
    own_session = session is None
    if session is None:
        session = _create_session()

    response = session.get(CIFP_URL, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    print("Page retrieved successfully.")

    try:
        print("Parsing HTML content...")
        # CIFP zip file names embed the AIRAC cycle date, e.g. CIFP_250123.zip
        # for the 2025-01-23 cycle, so the latest release is the one with the
        # maximum cycle date. The six-digit cycles are zero-padded, so they
        # sort identically as strings and as dates.
        cifp_zips = []
        for name, _ in _fetch_zip_links(CIFP_URL, response.text):
            match = CIFP_ZIP_NAME_PATTERN.match(name)
            if match is not None:
                cifp_zips.append((match.group(1), name))
        if not cifp_zips:
            raise ValueError("No CIFP zip links found")

        latest_cycle, latest_zip_file = max(cifp_zips)
        print(f"Latest CIFP file found: {latest_zip_file}")
        return latest_cycle
    finally:
        if own_session:
            session.close()


def _download_file(
    zip_url: str,
    download_folder: str,
    session: Optional[requests.Session] = None,
):
    print(f"Downloading: {zip_url}")

    def _download_with_session(download_session: requests.Session):
        zip_response = download_session.get(
            zip_url,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
        )
        zip_response.raise_for_status()

        zip_name = os.path.join(download_folder, os.path.basename(zip_url))
        with open(zip_name, "wb") as file:
            for chunk in zip_response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                file.write(chunk)
        return zip_name

    if session is not None:
        return _download_with_session(session)

    with _create_session() as download_session:
        return _download_with_session(download_session)


def download_cifp_zip(
    release_number: str,
    download_folder: str,
    session: Optional[requests.Session] = None,
):
    latest_zip_link = urljoin(CIFP_URL, f"CIFP_{release_number}.zip")

    # Download the latest zip file
    filename = os.path.join(download_folder, os.path.basename(latest_zip_link))
    os.makedirs(download_folder, exist_ok=True)

    own_session = session is None
    if session is None:
        session = _create_session()

    try:
        print(f"Downloading {latest_zip_link}...")
        zip_response = session.get(
            latest_zip_link, stream=True, timeout=DEFAULT_TIMEOUT
        )
        zip_response.raise_for_status()
        with open(filename, "wb") as file:
            for chunk in zip_response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    finally:
        if own_session:
            session.close()


def download_dtpp_zips(
    release_number: str,
    download_folder: str,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    session: Optional[requests.Session] = None,
):
    print(f"Fetching page: {DTPP_URL}")
    own_session = session is None
    if session is None:
        session = _create_session()

    response = session.get(DTPP_URL, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    print("Page retrieved successfully.")

    try:
        print("Parsing HTML content...")
        os.makedirs(download_folder, exist_ok=True)

        # Extract all .zip links involving the current release number
        zip_links = sorted(
            {
                zip_url
                for name, zip_url in _fetch_zip_links(DTPP_URL, response.text)
                if release_number in name
            }
        )
        if not zip_links:
            print(f"No DTPP zip files found for release {release_number}.")
            return []

        if max_workers <= 1:
            return [
                _download_file(
                    zip_url=zip_url,
                    download_folder=download_folder,
                    session=session,
                )
                for zip_url in zip_links
            ]

        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(zip_links))
        ) as executor:
            return list(
                executor.map(
                    lambda zip_url: _download_file(
                        zip_url=zip_url,
                        download_folder=download_folder,
                    ),
                    zip_links,
                )
            )
    finally:
        if own_session:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-folder",
        default="download",
        help="Directory to write downloaded zips into.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Number of worker threads for DTPP zip downloads.",
    )
    args = parser.parse_args()

    latest_release = get_latest_release_number()

    # Set the release number as an output if we're running on CI.
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"release={latest_release}\n")

    with _create_session() as session:
        download_cifp_zip(latest_release, args.download_folder, session=session)
        download_dtpp_zips(
            latest_release,
            args.download_folder,
            max_workers=max(1, args.workers),
            session=session,
        )
