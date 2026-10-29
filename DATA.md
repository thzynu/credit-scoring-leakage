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

Downloaded from the UCI Machine Learning Repository (CC BY 4.0, checked on the dataset pages).
The target column was saved as `target`. German uses labels 1/2 (2 = bad); the code sets
`positive=2` for it.

- German (UCI id 144): Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5NC77
- Australian (UCI id 143): Quinlan, R. Statlog (Australian Credit Approval). UCI Machine Learning
  Repository. https://doi.org/10.24432/C59012
- Taiwan (UCI id 350): Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C55S3H
  Introductory paper: Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card clients. Expert Systems with
  Applications.

## Lending Club (sample)

`lendingclub_100k.csv` is a 100,001-row sample of the granting-model version published on Zenodo:

Ariza-Garzon, M. J., Sanz-Guerrero, M., Arroyo Gallardo, J., & Lending Club. (2024).
Lending Club loan dataset for granting models (Version 0.1) [Dataset]. Universidad Complutense de
Madrid. https://doi.org/10.5281/zenodo.11295916 (CC BY 4.0)

- Original file: `LC_loans_granting_model_dataset.csv`, 167.5 MB, MD5 `b019384d6bc65bf2a3e839362e4ff502`
  (as listed on Zenodo). The Zenodo description gives 1,347,681 records.
- Sample in this repo: 100,001 rows, MD5 `cf57d101e3e9cbb0a6e5d0ec4ce92774`. The column `year` was
  derived from `issue_d`. Columns `desc` and `title` were dropped.
- Verification (run 2026-10-29 against the file downloaded from Zenodo, joined on `id`):
  all 100,001 sample rows are found in the original file; `loan_amnt` and `Default` match in
  100% of rows. The share of loans per issue year is identical to the original to four decimals
  in all 12 years (2007-2018), which is consistent with a sample stratified by issue year.
  The default rate is 0.2006 in the sample versus 0.1998 in the original, so the sample was
  not forced to keep the default rate.
- The sampling script and random seed were not recorded. The stratification by year is
  inferred from the numbers above, not from a script.

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
