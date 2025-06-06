# -*- coding: utf-8 -*-
"""Main entry for lottery utilities.

This script can update lottery draw data and also contains legacy
analysis code. Use ``--update`` with ``--type`` to append the latest
results to Google Sheets.
"""

import argparse
from lotterypython.update_data import main as update_lottery_data

def _legacy_analysis():
    """Placeholder for the original analysis code."""
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Lottery utilities")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch latest draws and append to Google Sheets",
    )
    parser.add_argument(
        "--type",
        choices=["big", "super"],
        default="big",
        help="Lottery type: big (lotto649) or super (superlotto638)",
    )
    args = parser.parse_args()

    if args.update:
        update_lottery_data(args.type)
    else:
        _legacy_analysis()


if __name__ == "__main__":
    main()
