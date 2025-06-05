# LotteryPython

This project fetches Taiwan lottery results and stores them in Google Sheets.

## Requirements
- Python 3.9+
- `gspread`, `oauth2client`, `cloudscraper`, `beautifulsoup4`
- A Google service account JSON placed as `credentials.json`

Install dependencies using `pip`:

```bash
pip install gspread oauth2client cloudscraper beautifulsoup4 requests
```

## Updating Data

Use `update_data.py` to append the latest draws to Google Sheets. Specify the lottery type with `--type`:

```bash
python update_data.py --type big    # Update 大樂透 (lotto649)
python update_data.py --type super  # Update 威力彩 (superlotto638)
```

The script uses `taiwan_lottery.py` to fetch the official history pages, parse the draw period, date, numbers, and special number, then append the results to the appropriate worksheets.

Each time data is fetched, the new rows are also appended to local CSV files
named `<type>_sequence.csv` and `<type>_sorted.csv` (where `<type>` is `big` or
`super`). These files mirror the two worksheets so you can keep a local archive
of the draw history.

