# LotteryPython

This project fetches Taiwan lottery results and stores them in Google Sheets.

## Requirements
- Python 3.9+
- `gspread`, `oauth2client`, `cloudscraper`, `beautifulsoup4`
- A Google service account JSON placed as `credentials.json` (provide separately)

## Environment Setup
Install the required Python packages using `pip` or `poetry`:

```bash
pip install gspread oauth2client cloudscraper beautifulsoup4 requests
# or
poetry add gspread oauth2client cloudscraper beautifulsoup4 requests && poetry install
```

Place your Google service account JSON in the project root as `credentials.json`. This file is not included in the repository.

## Updating Data

Use `update_data.py` to append the latest draws to Google Sheets. Specify the lottery type with `--type`:

```bash
python update_data.py --type big    # Update 大樂透 (lotto649)
python update_data.py --type super  # Update 威力彩 (superlotto638)
```

You can also run the same update via `main.py`:

```bash
python main.py --update --type big
```

`update_data.py` determines the request range automatically. It sets `start` to
the day after the last stored draw and `end` to today's date when querying
`lot539.com`.

The script uses `taiwan_lottery.py` to fetch draw results from [lot539.com](https://www.lot539.com), parse the draw period, date, numbers and special number, then append the results to the appropriate worksheets. Draws are sorted by period in ascending order before writing so that IDs and periods grow chronologically.

Each time data is fetched, the new rows are also appended to local CSV files
named `<type>_sequence.csv` and `<type>_sorted.csv` (where `<type>` is `big` or
`super`). These files mirror the two worksheets so you can keep a local archive
of the draw history.

The period value stored in the CSV files and Google Sheets is a six-digit
string. Any leading `'` character from the original draw period is removed and
the value is reformatted using `period[:3] + period[-3:]` before being
appended.

