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

To save prediction results from the built-in analysis tools, omit
`--update` and pass `--save-results`:

```bash
python -m lotterypython --type big --save-results
```

The results are appended to the `分析結果` worksheet of the same Google Sheet.

## macOS Scheduling Script

For automated updates on macOS, a helper script is available at
`scripts/run_lottery_schedule.sh`. The script checks the current weekday and
runs the appropriate commands:

- Every Tuesday and Friday it executes:
  - `python -m lotterypython --update --type big`
  - `python -m lotterypython --type big --save-results`
- Every Monday and Thursday it executes:
  - `python -m lotterypython --update --type super`
  - `python -m lotterypython --type super --save-results`

To have these commands run automatically, schedule the script with `cron` or
`launchd`. For example, to run the script daily at 9 AM with `cron`:

```bash
crontab -e
```

Add the following line (adjusting the path to the repository as needed):

```cron
0 9 * * 1-5 /usr/bin/env bash /path/to/LotteryPython/scripts/run_lottery_schedule.sh >> ~/Library/Logs/lotterypython.log 2>&1
```

The script skips days without scheduled draws, so running it Monday through
Friday is safe. Set the `PYTHON_BIN` environment variable before the command if
you need to use a specific Python interpreter.

## LSTM Analysis

An experimental script is provided to generate number predictions using an
LSTM neural network. It trains on the historical draws in
`lotterypython/big_sequence.csv` and prints a suggested set of numbers:

```bash
python predict/lotto_predict_lstm.py
```

The model is very small and intended only as a demonstration, so the output
should not be considered accurate.

## Docker

A `Dockerfile` and `docker-compose.yml` are included for running the web application with Nginx.

### Prerequisites

1. Ensure you have Docker and Docker Compose installed.
2. Place your Google service account JSON as `credentials.json` in the root directory.

### Running with Docker Compose

To start the web application and Nginx server:

```bash
docker-compose up -d --build
```

The application will be accessible at `http://localhost`.

### Running Manually with Docker

If you prefer to run only the application container:

Build the image:

```bash
docker build -t lottery-python .
```

Run the container (exposing port 3000):

```bash
docker run --rm -p 3000:3000 -v $(pwd)/credentials.json:/app/credentials.json lottery-python
```

Access the application at `http://localhost:3000`.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Predictions
| algorithm | numbers | special | tally |
|-----------|---------|---------|-------|
| RandomForest | 9 24 32 40 41 49 | 27 | 222 |
| GradientBoosting | 21 24 27 29 32 43 | 27 | 203 |
| KNN | 11 24 26 29 40 48 | 27 | 205 |
