from collections import Counter


def predict_hot50(df, today_index):
    """Predict using the hottest 50 draws."""
    train = df.iloc[today_index - 50:today_index]
    nums = train[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].values.ravel()
    counter = Counter(nums)
    top_10 = counter.most_common(10)
    cnt = counter.most_common(6)
    main = [int(n) for n, _ in cnt]
    special = train['Special'].value_counts().idxmax()
    special = int(special)

    # Return details with top 10 and their frequency counts
    details = {
        "type": "frequency_ranking",
        "top_10": [[int(n), int(freq)] for n, freq in top_10]
    }
    return main, special, details
