from collections import Counter

def predict_hot50(df, today_index):
    train = df.iloc[today_index-50:today_index]      # 最近 50 期
    nums  = train[['First','Second','Third','Fourth','Fifth','Sixth']].values.ravel()
    cnt   = Counter(nums).most_common(6)
    main  = [n for n,_ in cnt]                       # 6 個主號
    special = train['Special'].value_counts().idxmax()
    return main, special