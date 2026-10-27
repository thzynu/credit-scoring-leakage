import exp as E
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


def make_prep_q(num_cols, cat_cols, min_freq):
    parts = []
    if num_cols:
        num = SkPipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", QuantileTransformer(output_distribution="normal",
                                                     n_quantiles=200, random_state=0))])
        parts.append(("num", num, num_cols))
    if cat_cols:
        cat = SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=min_freq,
                                               sparse_output=False))])
        parts.append(("cat", cat, cat_cols))
    return ColumnTransformer(parts, sparse_threshold=0)


def apply():
    E.make_prep = make_prep_q
    E.DATASETS = {k + "_q": v for k, v in E.DATASETS.items()}
