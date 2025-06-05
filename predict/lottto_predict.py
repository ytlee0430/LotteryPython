import pandas as pd, numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

CSV_FILE = "../power_lottery.csv"  
SEQ_LEN  = 10

df = pd.read_csv(CSV_FILE)
df["Date"] = pd.to_datetime(df["Date"])

def encode(nums):
    v = np.zeros(49, dtype=np.float32)
    v[[n-1 for n in nums]] = 1.
    return v

def consec(nums):
    s = sorted(nums)
    return sum(1 for x,y in zip(s, s[1:]) if y-x == 1)

hot_sets = []
for i in range(len(df)):
    past = df.loc[max(0,i-50):i-1, ["First","Second","Third","Fourth","Fifth","Sixth"]].values.flatten()
    hot_sets.append(set([n for n,_ in Counter(past).most_common(10)]))

feats = []
for i,row in df.iterrows():
    nums = row[["First","Second","Third","Fourth","Fifth","Sixth"]].tolist()
    f = np.concatenate([
        encode(nums),
        np.eye(7)[row["Date"].weekday()],
        np.eye(6)[consec(nums)],
        [sum(1 for n in nums if n in hot_sets[i])]
    ])
    feats.append(f)
feats = np.stack(feats)

X, y = [], []
for i in range(len(feats) - SEQ_LEN):
    X.append(feats[i:i+SEQ_LEN])
    nxt_nums = df.loc[i+SEQ_LEN, ["First","Second","Third","Fourth","Fifth","Sixth"]].tolist()
    y.append(encode(nxt_nums))
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.9)
Xtr, Xv = X[:split], X[split:]
ytr, yv = y[:split], y[split:]

model = models.Sequential([
    layers.Input(shape=Xtr.shape[1:]),
    layers.LSTM(64),
    layers.Dense(128, activation='relu'),
    layers.Dense(49, activation='sigmoid')
])
model.compile('adam', 'binary_crossentropy')
model.fit(Xtr, ytr, epochs=25, batch_size=32, validation_data=(Xv, yv), verbose=0)

latest_seq = feats[:SEQ_LEN][np.newaxis, ...]
probs = model.predict(latest_seq, verbose=0)[0]
main6 = np.argsort(probs)[-6:][::-1] + 1

X_spec = feats[:-1]
y_spec = df["Special"].values[1:]
Xtr_s, Xv_s, ytr_s, yv_s = train_test_split(X_spec, y_spec, test_size=0.1, random_state=42)
rf_spec = RandomForestClassifier(n_estimators=200, random_state=42)
rf_spec.fit(Xtr_s, ytr_s)
special = rf_spec.predict(feats[0].reshape(1,-1))[0]

print("\n===== AI result =====")
print("numbers :", sorted(main6))
print("sp  :", int(special))
print("=======================")
