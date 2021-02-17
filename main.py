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


# type = 1 大樂透， type=2 威力彩
# http://www.9800.com.tw/statistics.asp 一般順
# http://www.9800.com.tw/drop.asp 落球順
dropTypeAddressDict = {"一般順": "http://www.9800.com.tw/statistics.asp", "落球順": "http://www.9800.com.tw/drop.asp"}
lotteryTypeAndTitleDict = {1: "big-lottery", 2: "power-lottery"}
isNeedUpdate = False

lotteryType = 2
dropType = "一般順"
address = dropTypeAddressDict[dropType]
isShowPlot = False
sampleCount = -1

isSpecial = False
predictNumber = 4 if not isSpecial else 1
columnStart = 3 if not isSpecial else 9
columnEnd = 9 if not isSpecial else 10

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ").worksheet(lotteryTypeAndTitleDict[lotteryType])

if isNeedUpdate:
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

if isShowPlot:
    dataset.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    pyplot.show()

    dataset.hist()
    pyplot.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()

# Split-out validation dataset
array = dataset.values
startIndex = len(array)-sampleCount
if sampleCount < 0 or startIndex < 0:
    startIndex = 0
y = []
X = array[startIndex:len(array) - 1, columnStart:columnEnd]
for nums in array[startIndex+1:, columnStart:columnEnd]:
    y.append(nums[predictNumber-1])

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

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
    # Make predictions on validation dataset 
    model.fit(X_train, Y_train)
    # predictions = model.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions)) 
    # print(confusion_matrix(Y_validation, predictions)) 
    # print(classification_report(Y_validation, predictions))
    if isSpecial:
        predictions = model.predict(array[len(array)-1:len(array), 9:10])
    else:
        predictions = model.predict(array[len(array)-1:len(array), 3:9])
    print(predictions)
