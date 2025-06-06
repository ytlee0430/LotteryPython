from pathlib import Path
import sys
import pytest

# Ensure project root is importable when tests are run from within the tests
# directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from taiwan_lottery import TaiwanLottery


@pytest.fixture
def sample_html():
    path = Path(__file__).resolve().parents[1] / "example.xml"
    return path.read_text(encoding="utf-8")


def test_get_latest_draws_parses_results(monkeypatch, sample_html):
    tl = TaiwanLottery()

    def fake_fetch_html(self, lottery_type, start=None, end=None):
        return sample_html

    monkeypatch.setattr(TaiwanLottery, "fetch_html", fake_fetch_html)

    draws = tl.get_latest_draws("big")
    assert draws, "No draws were parsed"

    first = draws[0]
    assert first.period == "113000020"
    assert first.date == "2024-03-07"
    assert first.numbers == ["20", "17", "11", "27", "13", "03"]
    assert first.special == "02"
