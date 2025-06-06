"""Parsing helpers for lottery HTML pages."""

from bs4 import BeautifulSoup

from lottery_data import LotteryData


class Parser:
    """Utility class to parse lottery draw HTML."""

    def parse_html(
        self,
        html: str,
        data: LotteryData,
        record_id: int,
        latest_period: int,
    ) -> LotteryData:
        """Parse draw tables and append new rows to ``data``.

        Parameters
        ----------
        html:
            Raw HTML snippet containing draw tables.
        data:
            ``LotteryData`` instance to append rows to.
        record_id:
            Numerical ID of the last inserted record.
        latest_period:
            Most recent draw period already processed.

        Returns
        -------
        LotteryData
            The updated ``LotteryData`` with new rows appended.
        """

        html = html.replace("\r\n", " ").replace("\n", " ").replace("<br>", " ")
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")

        for table in reversed(tables):
            cells = table.get_text(" ").split(" ")
            if len(cells) != 30:
                continue
            if int(cells[3]) <= latest_period:
                continue

            latest_period = int(cells[3])
            period = cells[3][:3] + cells[3][-3:]
            record_id += 1
            data.sorted_data.append([record_id, period, cells[2]] + cells[7:14])
            data.sequence_data.append([record_id, period, cells[2]] + cells[19:26])

        return data
