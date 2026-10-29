"""Rebuild data/lendingclub_100k.csv from the Zenodo file using the saved id list.
The original sampling seed was not recorded; the id list is the ground truth of the sample."""
import hashlib
import pandas as pd

SRC = "data/LC_loans_granting_model_dataset.csv"   # Zenodo 10.5281/zenodo.11295916
COLS = ["id", "issue_d", "revenue", "dti_n", "loan_amnt", "fico_n", "experience_c", "emp_length",
        "purpose", "home_ownership_n", "addr_state", "zip_code", "Default"]
ids = pd.read_csv("data/lendingclub_ids.csv")["id"]
df = pd.read_csv(SRC, usecols=COLS).set_index("id").loc[ids].reset_index()
df["year"] = df["issue_d"].str[-4:].astype(int)
df.to_csv("data/lendingclub_100k_rebuilt.csv", index=False)
print("rows:", len(df))
print("md5 rebuilt:", hashlib.md5(open("data/lendingclub_100k_rebuilt.csv", "rb").read()).hexdigest())
print("md5 current:", hashlib.md5(open("data/lendingclub_100k.csv", "rb").read()).hexdigest())
pd.testing.assert_frame_equal(df, pd.read_csv("data/lendingclub_100k.csv"), check_dtype=False)
print("OK: rebuilt sample equals the file used in the experiments")
