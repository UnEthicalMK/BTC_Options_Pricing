import os
import re
import glob
from datetime import datetime
from tardis_dev import download_datasets
from src import config

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def fetch_historical_deribit_data(start_date: str, end_date: str) -> list[str]:
    """
    Downloads historical Deribit options_chain data from Tardis.dev.

    Parameters
    ----------
    start_date : str
        Inclusive start date in 'YYYY-MM-DD' format.
    end_date : str
        Exclusive end date in 'YYYY-MM-DD' format (Tardis convention).

    Returns
    -------
    list[str]
        Absolute paths of every file written to RAW_DATA_DIR.

    Notes
    -----
    - Without a TARDIS_API_KEY env var, only the first calendar day of each
      month is accessible (Tardis free-tier restriction).
      Set the variable before calling:
          export TARDIS_API_KEY="your_key_here"
    - end_date is NON-INCLUSIVE: to download Jan 1 only, pass
      start_date="2024-01-01", end_date="2024-01-02".
    """
    # ── 1. Date format validation ──────────────────────────────────────────
    for label, d in [('start_date', start_date), ('end_date', end_date)]:
        if not _DATE_RE.match(d):
            raise ValueError(
                f"{label}='{d}' is not in required 'YYYY-MM-DD' format."
            )
        try:
            datetime.strptime(d, '%Y-%m-%d')
        except ValueError as exc:
            raise ValueError(f"{label}='{d}' is not a valid calendar date.") from exc

    if start_date >= end_date:
        raise ValueError(
            f"start_date ('{start_date}') must be strictly before end_date ('{end_date}')."
        )

    # ── 2. API key hint ────────────────────────────────────────────────────
    if not os.environ.get('TARDIS_API_KEY'):
        print(
            "  [data_fetcher] WARNING: TARDIS_API_KEY not set. "
            "Only the first day of each calendar month is available without an API key."
        )

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    # Record existing files so we can identify what was newly written.
    files_before = set(glob.glob(os.path.join(config.RAW_DATA_DIR, '*.csv.gz')))

    print(f"  [data_fetcher] Downloading Deribit options_chain "
          f"{start_date} → {end_date} (end exclusive)...")

    # ── 3. Download with error handling ───────────────────────────────────
    try:
        download_datasets(
            exchange="deribit",
            data_types=["options_chain"],
            symbols=["OPTIONS"],
            from_date=start_date,
            to_date=end_date,
            download_dir=config.RAW_DATA_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Tardis download failed for {start_date}–{end_date}. "
            f"Check your TARDIS_API_KEY and network connectivity. "
            f"Original error: {exc}"
        ) from exc

    # ── 4. Post-download file verification ────────────────────────────────
    files_after  = set(glob.glob(os.path.join(config.RAW_DATA_DIR, '*.csv.gz')))
    new_files    = sorted(files_after - files_before)

    if not new_files:
        raise RuntimeError(
            f"Tardis reported success but no new .csv.gz files were written to "
            f"{config.RAW_DATA_DIR}. The date range ({start_date}–{end_date}) may "
            f"require a paid API key, or may not contain options_chain data."
        )

    print(f"  [data_fetcher] Download complete. {len(new_files)} file(s) written:")
    for f in new_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"    {os.path.basename(f)}  ({size_mb:.1f} MB)")

    return new_files