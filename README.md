# LotteryPython

This project fetches Taiwan lottery results and stores them in Google Sheets.

## Requirements
- Python 3.9+
- `gspread`, `oauth2client`, `cloudscraper`, `beautifulsoup4`
- A Google service account JSON placed as `credentials.json`

Ensure `pip` is available. Most Python installations ship with it, and you can
upgrade to the latest version using:

```bash
python3 -m pip install --upgrade pip
```

Install dependencies using `pip`:

```bash
pip install gspread oauth2client cloudscraper beautifulsoup4 requests
```

## Updating Data

Use the package to append the latest draws to Google Sheets. Specify the lottery type with `--type`:

```bash
python -m lotterypython --update --type big    # Update 大樂透 (lotto649)
python -m lotterypython --update --type super  # Update 威力彩 (superlotto638)
```

`lotterypython.update_data` determines the request range automatically. It sets `start` to
the day after the last stored draw and `end` to today's date when querying
`lot539.com`.

The script uses `lotterypython.taiwan_lottery` to fetch draw results from [lot539.com](https://www.lot539.com), parse the draw period, date, numbers and special number, then append the results to the appropriate worksheets. Draws are sorted by period in ascending order before writing so that IDs and periods grow chronologically.

Each time data is fetched, the new rows are also appended to local CSV files
named `<type>_sequence.csv` and `<type>_sorted.csv` (where `<type>` is `big` or
`super`). These files mirror the two worksheets so you can keep a local archive
of the draw history.

The period value stored in the CSV files and Google Sheets is a six-digit
string. Any leading `'` character from the original draw period is removed and
the value is reformatted using `period[:3] + period[-3:]` before being
appended.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
