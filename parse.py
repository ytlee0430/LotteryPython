from bs4 import BeautifulSoup
from lottery_data import LotteryData

class Parser:
        def parse_html(self, html, data: LotteryData, id, latest_period):
                html = html.replace("\r\n", " ")
                html = html.replace("\n", " ")
                html = html.replace("<br>", " ")
                soup = BeautifulSoup(html)
                tables = soup.find_all("table")
                
                for t in enumerate(reversed(tables)):
                        d = t[1].text.split(" ")
                        if len(d) == 30 and int(d[3]) > latest_period:
                                latest_period = int(d[3])
                                period = d[3][:3] +  d[3][len(d[3])-3:]
                                id = id + 1
                                data.sorted_data.append([id, period, d[2]] + d[7:14])
                                data.sequence_data.append([id, period, d[2]] + d[19:26])
                        
                return data
