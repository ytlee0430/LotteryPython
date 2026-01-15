from collections import Counter


def predict_hot50(df, today_index):
    """Predict using the hottest 50 draws."""
    train = df.iloc[today_index - 50:today_index]
    nums = train[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].values.ravel()
    cnt = Counter(nums).most_common(6)
    main = [int(n) for n, _ in cnt]
    special = train['Special'].value_counts().idxmax()
    special = int(special)
    return main, special
