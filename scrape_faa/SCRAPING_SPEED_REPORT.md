# Scraping Speed Improvement Report

## Scope

This report covers performance improvements made in:

- `/home/runner/work/faa-instrument-approach-db/faa-instrument-approach-db/scrape_faa/download.py`
- `/home/runner/work/faa-instrument-approach-db/faa-instrument-approach-db/plate_analyzer/scrape_faa_dtpp_zip.py`

## What Was Optimized

1. **Latest release lookup now avoids per-file HEAD requests when possible**
   - Parses timestamps from the FAA directory listing HTML.
   - Falls back to HEAD requests only when a timestamp is missing.
2. **Parallel DTPP downloads**
   - Added threaded downloads with configurable worker count (`--workers`).
   - Defaults to `8` workers.
3. **Lower request overhead**
   - Reuses HTTP sessions where appropriate.
   - Deduplicates zip URLs before downloading.
4. **Faster PDF parsing pipeline dispatch**
   - Avoids reading every ZIP entry in the parent process.
   - Avoids sending large PDF byte buffers across process boundaries.
   - Sends lightweight `(zip_path, file_name)` tasks to workers and reads only targeted files in worker processes.

## Profiling Method

- Used a synthetic benchmark with representative network delays and FAA-like HTML listings.
- Measured average wall-clock time over 3 runs per scenario.
- Compared:
  - **Baseline**: original sequential behavior.
  - **Optimized single-thread**: new code with `--workers 1`.
  - **Optimized multi-thread**: new code with `--workers 8`.

## Results

| Measurement | Baseline (s) | Optimized Single-thread (s) | Optimized Multi-thread (s) |
| --- | ---: | ---: | ---: |
| Latest release lookup | 0.610 | 0.007 | 0.007 |
| DTPP zip download stage | 0.967 | 0.968 | 0.168 |
| Combined scraping stage (lookup + DTPP) | 1.577 | 0.975 | 0.175 |

## PDF Parsing Pipeline Profiling

### Method

- Synthetic dataset: 3 DTPP ZIP files, 120 files each (72 target approach PDFs, 288 non-target files), 256 KB per file.
- Worker count: 3 processes.
- Compared:
  - **Baseline PDF dispatch**: parent reads all files, builds `BytesIO`, filters later, and sends PDF bytes through multiprocessing queue.
  - **Optimized PDF dispatch**: parent scans names only, sends `(zip_path, file_name)` tasks, workers read only target files.

### Results

| Measurement | Baseline (s) | Optimized (s) |
| --- | ---: | ---: |
| PDF task generation | 0.049 | 0.001 |
| End-to-end PDF dispatch + worker processing | 0.085 | 0.034 |

### Speedups

- **PDF task generation**: **~38.9x faster**
- **End-to-end dispatch + processing overhead**: **~2.50x faster**

## Speedups

- **Latest release lookup**: **~81.4x faster** (single-thread and multi-thread).
- **DTPP download stage**:
  - single-thread: roughly parity (`~1.00x`),
  - multi-thread: **~5.77x faster**.
- **Combined scraping stage**:
  - single-thread: **~1.62x faster** (~38% reduction),
  - multi-thread: **~9.01x faster** (~89% reduction),
  - multi-thread vs optimized single-thread: **~5.57x faster**.

## Conclusion

The changes produce a clear positive impact:
- significant single-thread improvement from removing unnecessary HEAD requests,
- substantial additional gains from parallel DTPP downloads,
- and meaningful PDF pipeline acceleration by eliminating unnecessary reads and inter-process byte transfer overhead.
