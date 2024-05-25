from bs4 import BeautifulSoup
from lottery_data import LotteryData

class Parser:
        def parse_html(self, html, data: LotteryData, id, latest_period):
                html = html.replace("\r\n", " ")
                html = html.replace("\n", " ")
                html = html.replace("<br>", " ")
                soup = BeautifulSoup(html)
                tables = soup.find_all("table")
                
                for i, t in enumerate(reversed(tables)):
                        d = t.text.split(" ")
                        if len(d) == 30 and int(d[3]) > latest_period:
                                latest_period = int(d[3])
                                period = d[3][:3] +  d[3][len(d[3])-3:]
                                id = id + 1
                                sorted_row = {'ID': id, 'Period': period, 'Date':d[2], 'First': d[7], 'Second': d[8], 'Third': d[9], 'Fourth': d[10], 'Fifth': d[11], 'Sixth': d[12], 'Special': d[13]}
                                data.sorted_data.append([id, period, d[2]] + d[7:14])
                                sequence_row =  {'ID': id, 'Period': period, 'Date':d[2], 'First': d[19], 'Second': d[20], 'Third': d[21], 'Fourth': d[22], 'Fifth': d[23], 'Sixth': d[24], 'Special': d[25]}
                                data.sequence_data.append([id, period, d[2]] + d[19:26])
                        
                return data

# p = Parser()
# data = LotteryData()
# html = open("example.xml", "r").read()
# p.parse_html(html, data)
# print(data)