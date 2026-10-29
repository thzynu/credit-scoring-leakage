# Data sources and licences

The code in `code/` expects five CSV files in `data/`. Four of them are redistributed in this
repository under CC BY 4.0. **`gmsc.csv` is not included** and must be downloaded by the reader.

| File | Rows | Source | Licence | In repo? |
|---|---|---|---|---|
| `german.csv` | 1,000 | UCI, Statlog (German Credit Data) | CC BY 4.0 | yes |
| `australian.csv` | 690 | UCI, Statlog (Australian Credit Approval) | CC BY 4.0 | yes |
| `taiwan.csv` | 30,000 | UCI, Default of Credit Card Clients | CC BY 4.0 | yes |
| `lendingclub_100k.csv` | 100,001 | Zenodo, Lending Club loan dataset for granting models (sample) | CC BY 4.0 | yes |
| `gmsc.csv` | 150,000 | Kaggle, Give Me Some Credit | competition terms (not read in full) | **no** |

## German, Australian, Taiwan (UCI)

Downloaded from the UCI Machine Learning Repository. The target column was saved as `target`.
German uses labels 1/2 (2 = bad); the code sets `positive=2` for it.

- German: UCI dataset id 144, https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Australian: UCI dataset id 143, https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval
  Quinlan, R. Statlog (Australian Credit Approval). UCI Machine Learning Repository.
  https://doi.org/10.24432/C59012 (CC BY 4.0)
- Taiwan: UCI dataset id 350, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

TODO before submission: copy the exact citation (authors, year, DOI) for German and Taiwan from the
"Cite" button on their UCI pages, and paste it here.

## Lending Club (sample)

`lendingclub_100k.csv` is a 100,001-row sample of the granting-model version published on Zenodo:

Ariza-Garzon, M. J., Sanz-Guerrero, M., Arroyo Gallardo, J., & Lending Club. (2024).
Lending Club loan dataset for granting models (Version 0.1) [Dataset]. Universidad Complutense de
Madrid. https://doi.org/10.5281/zenodo.11295916 (CC BY 4.0)

- Original file: `LC_loans_granting_model_dataset.csv`, 167.5 MB, MD5 `b019384d6bc65bf2a3e839362e4ff502`
  (as listed on Zenodo).
- Sample in this repo: MD5 `cf57d101e3e9cbb0a6e5d0ec4ce92774`. The column `year` was derived from `issue_d`.
  Columns `desc` and `title` were dropped.
- TODO: state how the sample was drawn (random seed, whether stratified by year and default rate)
  and confirm the MD5 of the original file you started from.

## Give Me Some Credit (not redistributed)

Source: Kaggle competition "Give Me Some Credit" (Credit Fusion, 2011),
https://www.kaggle.com/c/GiveMeSomeCredit

Why it is not in this repo: the competition rules page states that privately sharing data outside
teams is not permitted, and the full Data Access and Use terms were not read by the authors of this
repository. To stay on the safe side the file is not redistributed. Its history was removed from
git.

To reproduce:
1. Log in to Kaggle and accept the competition terms.
2. Download `cs-training.csv` from the competition Data tab.
3. Save it as `data/gmsc.csv`. The file used in the paper has 11 columns, starting with the
   target `SeriousDlqin2yrs` and 150,000 rows. If your download has an extra unnamed index column
   in front, drop it so the columns match.

TODO: after reading the full terms, paste the "Data Access and Use" paragraph here with the date read.
