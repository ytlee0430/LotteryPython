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
    BASE_URLS = {
        "big": "https://www.taiwanlottery.com.tw/lotto/lotto649/history.aspx",
        "super": "https://www.taiwanlottery.com.tw/lotto/superlotto638/history.aspx",
    }

    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    def fetch_html(self, lottery_type: str) -> str:
        url = self.BASE_URLS[lottery_type]
        res = self.scraper.get(url, headers=self.headers)
        res.raise_for_status()
        return res.text

    def parse_draws(self, html: str) -> List[Draw]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id=lambda x: x and "history" in x.lower())
        if not table:
            return []
        rows = table.find_all("tr")[2:]
        draws = []
        for row in rows:
            cols = [td.text.strip() for td in row.find_all("td") if td.text.strip()]
            if len(cols) < 8:
                continue
            period = cols[0]
            date = cols[1]
            numbers = cols[2:8]
            special = cols[8] if len(cols) > 8 else cols[-1]
            draws.append(Draw(period, date, numbers, special))
        return draws

    def get_latest_draws(self, lottery_type: str, count: int = 10) -> List[Draw]:
        html = self.fetch_html(lottery_type)
        draws = self.parse_draws(html)
        return draws[:count]
