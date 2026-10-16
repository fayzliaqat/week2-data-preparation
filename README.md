# Week 2: Loan Data Preparation

**Fayz Liaqat | 16 October 2026**

This assignment prepares a raw loan-application dataset for a machine learning model.
I explore the data, check duplicates and missing values, and create two simple features.
I make a stratified 80/20 split, then fill, encode, and scale inputs using training data only.
The final outputs are `X_train`, `X_test`, `y_train`, and `y_test`; no model is trained.

## Dataset

| Item | Details |
| --- | --- |
| Name | Loan Predication |
| Source | [Kaggle dataset by amit parjapat](https://www.kaggle.com/datasets/ninzaami/loan-predication) |
| Included file | `data/loan_data.csv` |
| Raw shape | 614 rows x 13 columns |
| Target | `Loan_Status`: Y = approved, N = not approved |
| Problem | Supervised binary classification |
| Data types | Numeric inputs and text categories |
| Data quality | Seven columns have missing values; zero exact duplicate rows |
| Kaggle license label | CC0: Public Domain |

The CSV is small, so it is included. It has real missing values; none were added. Its original bytes are preserved. The notebook renames columns to camelCase in memory and drops the application ID. The source description says 615 rows; the downloaded CSV contains a header and **614 data rows**.

## Notebook workflow

1. Load and explore: shape, `info()`, `head()`, `describe()`, and four EDA charts.
2. Clean: remove exact duplicates, inspect missing values, drop the ID, and explain each filling decision.
3. Engineer `totalIncome` and `hasCoapplicantIncome`.
4. Separate inputs `X` from the target `y`.
5. Split 80/20 with `random_state=42` and `stratify=y`.
6. Use a `ColumnTransformer` with numeric and categorical pipelines.
7. Check final shapes, missing values, row alignment, and training-only fitted statistics.
8. Save the four prepared outputs as CSV files.

Numeric columns use median imputation and `StandardScaler`. Text columns use most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`. The preprocessor calls `fit_transform` on training inputs and only `transform` on test inputs. Targets use a fixed mapping: N = 0 and Y = 1.

## How to run

Use **Python 3.11 or newer**. Download or clone this repository, open the project folder in a terminal, and run:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m notebook week2_data_preparation.ipynb
```

On macOS or Linux, use `venv/bin/python` in place of `.\venv\Scripts\python.exe`.

In Jupyter, choose **Kernel > Restart Kernel and Run All Cells**. Keep the notebook in the project root so it can find `data/loan_data.csv`. No Kaggle account, API key, GPU, or new dataset download is needed. Internet is needed only to install packages.

The notebook includes executed results and can also be read directly on GitHub.

## Final outputs

| Variable | Shape | Saved file |
| --- | --- | --- |
| `X_train` | (491, 22) | [X_train.csv](outputs/X_train.csv) |
| `X_test` | (123, 22) | [X_test.csv](outputs/X_test.csv) |
| `y_train` | (491,) | [y_train.csv](outputs/y_train.csv) |
| `y_test` | (123,) | [y_test.csv](outputs/y_test.csv) |

The CSV target files have one column; loading them with `read_csv` produces a one-column DataFrame. Inside the notebook, `y_train` and `y_test` are one-dimensional Series as required.

The 22 input columns contain seven scaled numeric features and fifteen one-hot columns. One-hot columns stay 0/1. All four outputs contain zero missing values. Corresponding input and target CSVs preserve the same row order and omit the CSV index.

## Files to review

- [Completed notebook](week2_data_preparation.ipynb)
- [Three-page assignment report](report/Week_2_Data_Preparation_Report.pdf)
- [Dataset source notes](data/README.md)
- [Required packages](requirements.txt)
- `outputs/`: five charts and four prepared CSV files

## Notes

The code uses meaningful camelCase variable names. The four final names follow the assignment's required spelling. Missing values are intentionally left in the raw features until after splitting. No imputer, scaler, or encoder learns from the test set.

The added features are reasonable ideas, but their predictive benefit has not been measured. Standard scaling does not remove outliers or make skewed data normal. This assignment prepares data for learning; it does not evaluate a model or establish suitability for lending decisions.
