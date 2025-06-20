"""Update lottery results to Google Sheets."""

from datetime import datetime, timedelta
import argparse
import csv
from pathlib import Path
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from .taiwan_lottery import TaiwanLottery
from lottery_data import LotteryData


def _save_to_local_csv(lotto_type: str, mode: str, rows: list[list]):
    """Append new rows to a local CSV file."""
    csv_name = f"{lotto_type}_{mode}.csv"
    path = Path(__file__).resolve().parent / csv_name
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def add_one_day(date_str, date_format='%Y-%m-%d'):
    # Convert the string to a datetime object
    date_obj = datetime.strptime(date_str, date_format)

    # Add one day
    next_day = date_obj + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")

# type=big 大樂透， type=super 威力彩
lotteryTypeAndTitleDict = {"big": "big-lottery", "super": "power-lottery"}
dropType = "一般順"

CSV_HEADER = [
    "ID",
    "Period",
    "Date",
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Special",
]


def _sync_csv_with_sheet(path: Path, records: list[dict]):
    """Ensure local CSV contains all rows from the given sheet records."""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(CSV_HEADER)
            for rec in records:
                writer.writerow([rec[h] for h in CSV_HEADER])
        return

    # Gather existing IDs in the CSV
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        existing = {row["ID"] for row in reader}

    missing = [rec for rec in records if str(rec["ID"]) not in existing]
    if missing:
        with path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            for rec in missing:
                writer.writerow([rec[h] for h in CSV_HEADER])


def main(lotto_type: str) -> None:
    tl = TaiwanLottery()

    # use creds to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds']
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    sequence_sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ")\
        .worksheet(lotteryTypeAndTitleDict[lotto_type]+"-"+"落球順")
    sorted_sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ")\
        .worksheet(lotteryTypeAndTitleDict[lotto_type]+"-"+"一般順")

    all_record_sequence = sequence_sheet.get_all_records()
    all_record_sorted = sorted_sheet.get_all_records()
    lottery_data = LotteryData(lotto_type, [], [])

    seq_path = Path(__file__).resolve().parent / f"{lotto_type}_sequence.csv"
    sort_path = Path(__file__).resolve().parent / f"{lotto_type}_sorted.csv"

    # Ensure local CSVs contain all records currently stored in Google Sheets
    _sync_csv_with_sheet(seq_path, all_record_sequence)
    _sync_csv_with_sheet(sort_path, all_record_sorted)

    if not all_record_sequence:
        latest_id = 0
        latest_period = 0
        start_date = None
    else:
        latest_record = max(all_record_sequence, key=lambda x: x['ID'])
        latest_id = int(latest_record['ID'])
        latest_period = int(latest_record['Period'])
        start_date = add_one_day(latest_record['Date'])

    end_date = datetime.today().strftime('%Y-%m-%d')

    draws = tl.get_latest_draws(lotto_type, start=start_date, end=end_date)
    # Ensure chronological order so IDs and periods increase over time
    draws.sort(key=lambda d: int(d.period))

    sequence_rows = []
    sorted_rows = []
    for draw in draws:
        period = draw.period.lstrip("'")
        period = period[:3] + period[-3:]
        try:
            period_num = int(period)
        except ValueError:
            continue
        if period_num <= latest_period:
            continue
        latest_id += 1
        nums = [int(n) for n in draw.numbers]
        special = int(draw.special)
        sequence_rows.append([latest_id, period, draw.date] + nums + [special])
        sorted_rows.append([latest_id, period, draw.date] + sorted(nums) + [special])


    if sequence_rows:
        sequence_sheet.append_rows(sequence_rows)
        sorted_sheet.append_rows(sorted_rows)
        _save_to_local_csv(lotto_type, "sequence", sequence_rows)
        _save_to_local_csv(lotto_type, "sorted", sorted_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update lottery data")
    parser.add_argument(
        "--type",
        choices=["big", "super"],
        default="big",
        help="Lottery type: big (lotto649) or super (superlotto638)",
    )
    args = parser.parse_args()
    main(args.type)

