# FAA Scraper

Utility script to scrape and download the latest d-TPP release from the FAA
website automatically.

This gets run by the CI to automatically generate releases.

## Usage

```bash
python download.py --workers 8
```

- `--workers` controls parallel DTPP zip downloads.
- `--download-folder` controls where files are written (default: `download`).
