import pandas as pd
for n in ["german", "australian", "gmsc", "lendingclub_100k"]:
    d = pd.read_csv(f"data/{n}.csv")
    print("=====", n, d.shape)
    print(d.dtypes.value_counts().to_dict())
    print(d.head(2).T.to_string())
