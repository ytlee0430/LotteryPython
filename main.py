from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from bs4 import BeautifulSoup
import requests

dropTypeAddressDict = {"一般順": "http://www.9800.com.tw/statistics.asp", "落球順": "http://www.9800.com.tw/drop.asp"}

dropType = "一般順"
address = dropTypeAddressDict[dropType]
isNeedUpdate = False
lotteryType = 1

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ").sheet1

if isNeedUpdate:
    # type = 1 大樂透， type=2 威力彩
    # http://www.9800.com.tw/statistics.asp 一般順
    # http://www.9800.com.tw/drop.asp 落球順
    d = {'p1': '092001', 'p2': '120000', 'l': 0, 'type': lotteryType}
    response = requests.post(address, data=d)
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find_all("table")[1]
    trs = table.find_all("tr")[2:]
    data = [['ID', 'Period', 'Date', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Special']]
    for i, tr in enumerate(trs):
        tds = tr.find_all("td")[:9]
        tds = [td.text.strip() for td in tds]
        data.append([i+1] + [ele for ele in tds if ele])
    dataset = DataFrame(data[1:], columns=data[0])
    sheet.update([dataset.columns.values.tolist()] + dataset.values.tolist())
#
# # Extract and print all of the values
list_of_hashes = sheet.get_all_values()
dataset = DataFrame(list_of_hashes[1:],
                    columns=list_of_hashes[0])
dataset['ID'] = dataset['ID'].astype(float)
dataset['Period'] = dataset['Period'].astype(float)
dataset['First'] = dataset['First'].astype(float)
dataset['Second'] = dataset['Second'].astype(float)
dataset['Third'] = dataset['Third'].astype(float)
dataset['Fourth'] = dataset['Fourth'].astype(float)
dataset['Fifth'] = dataset['Fifth'].astype(float)
dataset['Sixth'] = dataset['Sixth'].astype(float)
dataset['Special'] = dataset['Special'].astype(float)
# print(dataset)
#
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('ID').size())

dataset.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Split-out validation dataset
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

