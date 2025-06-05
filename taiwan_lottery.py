from dataclasses import dataclass
from typing import List
import cloudscraper
from bs4 import BeautifulSoup

@dataclass
class Draw:
    period: str
    date: str
    numbers: List[str]
    special: str

class TaiwanLottery:
    """Scrape draw results from lot539.com."""

    BASE_URLS = {
        "big": "https://www.lot539.com/lottery/search?type=big",
        "super": "https://www.lot539.com/lottery/search?type=super",
    }

    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    def fetch_html(
        self,
        lottery_type: str,
        start: str | None = None,
        end: str | None = None,
    ) -> str:
        """Retrieve raw HTML for the given lottery type and date range."""
        url = self.BASE_URLS[lottery_type]
        parts = []
        if start:
            parts.append(("start", start))
        if end:
            parts.append(("end", end))
        if parts:
            url += "&" + "&".join(f"{k}={v}" for k, v in parts)
        res = self.scraper.get(url, headers=self.headers)
        res.raise_for_status()
        return res.text

    def parse_draws(self, html: str) -> List[Draw]:
        """Parse draw information from lot539 search results."""
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.select("div.content table.table.is-bordered")
        draws = []
        for table in tables:
            first_row = table.find("tr")
            if not first_row:
                continue
            cells = first_row.find_all("td")
            if len(cells) < 2:
                continue
            period_date = cells[0].get_text("\n", strip=True).split("\n")
            if len(period_date) < 2:
                continue
            period, date = period_date[0], period_date[1]

            nums = []
            special = ""
            for p in cells[1].find_all("p"):
                if "落球順序" in p.get_text():
                    for span in p.find_all("span", class_="lottery-ball"):
                        text = span.text.strip()
                        if "is-special" in span.get("class", []):
                            special = text
                        else:
                            nums.append(text)
                    break
            if nums:
                draws.append(Draw(period, date, nums, special))
        return draws

    def get_latest_draws(
        self,
        lottery_type: str,
        *,
        start: str | None = None,
        end: str | None = None,
        count: int | None = None,
    ) -> List[Draw]:
        """Fetch and parse draw results within a date range."""
        html = self.fetch_html(lottery_type, start=start, end=end)
        draws = self.parse_draws(html)
        if count is not None:
            return draws[:count]
        return draws
