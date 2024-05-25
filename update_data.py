from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import httpx
import cloudscraper

from parse import Parser
from lottery_data import LotteryData

def add_one_day(date_str, date_format='%Y-%m-%d'):
    # Convert the string to a datetime object
    date_obj = datetime.strptime(date_str, date_format)

    # Add one day
    next_day = date_obj + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")

# type=big 大樂透， type=super 威力彩
lotteryTypeAndTitleDict = {"big": "big-lottery", "super": "power-lottery"}
dropType = "一般順"
type="big"



# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sequence_sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ")\
    .worksheet(lotteryTypeAndTitleDict[type]+"-"+"落球順")
sorted_sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ")\
    .worksheet(lotteryTypeAndTitleDict[type]+"-"+"一般順")
all_record_sequence = sequence_sheet.get_all_records()
all_record_sorted = sorted_sheet.get_all_records()
lottery_data = LotteryData(type, [], [])

latest_record = max(all_record_sequence, key=lambda x: x['ID'])
latest_date = latest_record['Date']
latest_id = latest_record['ID']
latest_period = latest_record['Period']

base_url = f"https://www.lot539.com/lottery/search?start={add_one_day(latest_date)}"
final_url = f"{base_url}&type={type}"
responsex = httpx.get(final_url)
scraper = cloudscraper.create_scraper(browser={'browser': 'firefox','platform': 'windows','mobile': False})
html = scraper.get(final_url).content.decode('utf-8')

parser = Parser()
parser.parse_html(html, lottery_data, latest_id, latest_period)
sequence_sheet.append_rows(lottery_data.sequence_data)
sorted_sheet.append_rows(lottery_data.sorted_data)
