# Engineered datasets: handoff guide for the data scientist and the analyst

From: data engineering. To: data scientist, analyst.

This folder holds the cleaned datasets of a reservoir (oil and gas field) project for the ExxonMobil DataWorks Challenge 2026. The project builds a fast model of reservoir simulations, matches it to the field's real production history, and forecasts production to 2028.

The data comes from the organiser's raw Excel files. Those were hard to use: very wide tables, mostly empty cells, and files of 330 MB each. I converted them once into clean tables (CSV) and arrays (NumPy). **You do not need the raw files or any other document.** The one exception is the forecast template `09 Template Deliverable.xlsx`, which is included in this folder, unchanged. This guide explains every file in this folder, what each column means, and which files to use.

Two words you will see often:
- **Case**: one simulation run of the same field with one set of 4 uncertain rock parameters. There are 100 cases.
- **History**: the real, observed production of the field from 1998 to 2008.

> Confidential. The files come from the DWC26 dataset. Do not publish them. Do not put them in the code zip.

## 1. Which datasets should you use?

You can do the whole project with five files.

| Tier | File | Use it for |
|---|---|---|
| **Core** | `cases.csv` | Inputs of the model: the 4 parameters of each case, and its split (train, validation, test) |
| **Core** | `curves_quarterly_cum.npz` | Targets: production curves of the 100 cases (`Y`) and the observed history (`y_hist`), as arrays |
| **Core** | `history_observed.csv` | The observed history in table form, with rates and water injection |
| **Core** | `forecast_dates.csv` | The 80 dates of the forecast (2008-04-01 to 2028-01-01) |
| **Core** | `09 Template Deliverable.xlsx` | The forecast template to fill. It is the organiser's original file, unchanged. |
| **Core** | `param_bounds.csv` | Search range of each parameter |
| Extra | `curves_quarterly.csv` | Same curves as the `.npz`, in table form, with rates and the split. Easier for plots and Power BI. |
| Extra | `curves_native.csv` | All simulator time steps (finer than quarterly) |
| Optional | `grid_case_features.csv`, `grid_cells_template.csv`, `grid_varying_cells.npy`, `grid_active_cell_ids.npy` | Only to test whether the grid adds anything. The 4 parameters already carry the same information. |
| Reference | `grid_column_catalog.csv`, `grid_case_checks.csv`, `MANIFEST.csv`, `cache/` | Documentation and checks. Do not model on them. |

> **Remark on `grid_cells_template.csv`:** this file is probably not useful for a standard model. It describes the parts of the grid that are the same in all 100 cases (cell positions, starting pressure, saturation limits and so on). A model that predicts the production curves from the 4 parameters gets no new information from it, because it never changes from case to case. It only becomes useful if the team decides to build a spatial model (for example a CNN or another model that reads the grid as a 3D map). In that case, use it together with `grid_varying_cells.npy`. Otherwise, you can ignore it.

Where to start:

| You are the | Start with |
|---|---|
| Data scientist (proxy, history matching, forecast) | `cases.csv` + `curves_quarterly_cum.npz`, then `history_observed.csv` and `param_bounds.csv`. Use `forecast_dates.csv` to fill the forecast template (`09 Template Deliverable.xlsx`). |
| Analyst (exploration, charts, dashboard) | `curves_quarterly.csv` (one table: case, date, curves, split), `history_observed.csv`, `cases.csv` |

## 2. Where each file comes from

```
 RAW FILE (organiser's Excel; only         CLEAN FILE (the files in this folder)
           file 09 is in this folder)

 01 Introduction ---------------------->  cases.csv
                                          param_bounds.csv  (from the train cases in cases.csv)

 02 tab "Field Production"
  + column RAMP_History ---------------->  history_observed.csv

 02 six curve tabs -------------------->  curves_native.csv
                                              |  keep the 41 history dates
                                              v
                                          curves_quarterly.csv  --->  curves_quarterly_cum.npz (Y)
 history_observed.csv ------------------> supplies the 41 dates   --->  curves_quarterly_cum.npz (y_hist)

 09 Template -------------------------->  forecast_dates.csv

 03-08 grid sheets (100 cases) -------->  grid_cells_template.csv
                                          grid_varying_cells.npy
                                          grid_active_cell_ids.npy
                                          grid_column_catalog.csv
                                          grid_case_checks.csv
 grid_varying_cells.npy --------------->  grid_case_features.csv
```
*Caption: the left side is the raw file, the right side is what was made from it. The quarterly files are cut from the native table.*

| File | Made from | Shape | What it is |
|---|---|---|---|
| `cases.csv` | `01` (split from the workbook that holds each case) | 100 rows | Parameters, split and fold of each case |
| `param_bounds.csv` | `cases.csv`, train cases only | 4 rows | Smallest and largest parameter value |
| `history_observed.csv` | `02`: tab `Field Production` and column `RAMP_History` | 41 rows | The observed field history |
| `curves_native.csv` | `02`: six curve tabs, reshaped to long format | 17,222 rows | Every simulator step of the 100 cases |
| `curves_quarterly.csv` | `curves_native.csv`, the 41 dates of `history_observed.csv`, and the split from `cases.csv` | 4,100 rows (100 x 41) | The curves on the history dates |
| `curves_quarterly_cum.npz` | `Y` from `curves_quarterly.csv`, `y_hist` from `history_observed.csv` | arrays | The cumulative curves as arrays |
| `forecast_dates.csv` | `09` | 80 rows | The forecast date axis |
| `grid_cells_template.csv` | `03` (case 1) | 90,365 rows | Grid columns that never change, stored once |
| `grid_varying_cells.npy` | `03`-`08`, all 100 sheets | 100 x 42,512 x 6 | Grid columns that change between cases |
| `grid_active_cell_ids.npy` | `03` (case 1) | 42,512 | Row numbers of the active cells in the template |
| `grid_case_features.csv` | `grid_varying_cells.npy` | 100 rows | Summary numbers of the grid, per case |
| `grid_column_catalog.csv` | `03`-`08` raw sheets | 29 rows | Name and role of each raw grid column |
| `grid_case_checks.csv` | `03`-`08` raw sheets | 100 rows | Active-cell check per case |

### What "active cell" means

We define a cell as active when its pore volume (`porv`) is greater than 0. The rule is in `solution/grid_loader.py:61`:

```python
active = arr[:, names.index("porv")] > 0
```

How the rule works:

```
 grid sheet (90,365 rows)
        |
        v
  porv > 0 ?
     /      \
   yes       no
    |         |
 active    inactive
 (42,512)  (47,853)
```
*Caption: one test per cell. The pore volume column decides everything.*

Why pore volume:

- Pore volume is the space in the rock that can hold fluid. Zero pore volume means the cell holds no fluid, so the simulator has nothing to compute there.
- The data supports this choice. Inactive cells have empty values in other columns, such as `permz` and `tranz`. The count of empty cells (47,853) matches the inactive count.
- The data files have no `ACTNUM` column. `ACTNUM` is the flag that simulators normally use for active cells. So we infer the status from `porv`.

### How the grid files split into "template" and "varying"

```
  100 grid sheets (cases 1-100), 29 columns each
        |
        |  compare every column across all 100 cases
        v
  +-------------------------+     +-----------------------------+
  | 23 columns: identical   |     |  6 columns: differ by case  |
  | in every case           |     |  porv, poro, permx, permy,  |
  | (max difference = 0)    |     |  tranx, trany               |
  +-------------------------+     +-----------------------------+
        |                                    |
        v                                    v
 grid_cells_template.csv            grid_varying_cells.npy
 (read from case 1 only,            (kept for each case)
  25 cols incl. cell_id, active)
```
*Caption: the template holds the part of the grid that never changes from case to case.*

Details:

- The data comes from the sheet of case 1 (`03 Train Cases 1-4.xlsx`). Because the values are identical in all 100 cases, any case would give the same result.
- Nothing is averaged or combined. A check on all 100 cases confirmed a maximum difference of 0 for these columns.
- The 25 columns in the CSV are 23 grid columns plus `cell_id` and `active`. I added those two when building the file.

## 3. What the `.npz` and `.npy` files contain

NumPy files hold arrays. They are smaller and faster than CSV, and they keep full precision.

| File | Contents | Used for |
|---|---|---|
| `curves_quarterly_cum.npz` | `Y`: cumulative gas, oil, water for 100 cases x 41 dates x 3 (order gas, oil, water). `y_hist`: the observed history, 41 x 3. Also `case_ids`, `dates`, `t_days`, `quantities`. | The training targets and the history-matching target, ready to use |
| `grid_varying_cells.npy` | The 6 grid columns that change between cases (`porv`, `poro`, `permx`, `permy`, `tranx`, `trany`), for the 42,512 active cells of each case. Shape 100 x 42,512 x 6. | Grid experiments, maps, or a CNN. Optional. |
| `grid_active_cell_ids.npy` | The row numbers of the 42,512 active cells in `grid_cells_template.csv` | Links the array above to the template |

Case index in every array = `case_id` - 1. Load the `.npz` with `np.load(path, allow_pickle=False)`.

### How to open and view the `.npz` file

`curves_quarterly_cum.npz` is a bundle of several arrays. The code below lists them, then shows them as tables with column names. Change `case_id` to look at another case.

```python
import numpy as np, pandas as pd
OUT = "./"                                   # the folder that holds these files

z = np.load(OUT + "curves_quarterly_cum.npz", allow_pickle=False)
print(z.files)                               # ['Y', 'y_hist', 'case_ids', 'dates', 't_days', 'quantities']
print(z["Y"].shape, z["y_hist"].shape)       # (100, 41, 3) (41, 3)

cols  = [str(q) for q in z["quantities"]]    # ['gas_cum', 'oil_cum', 'water_cum']
dates = z["dates"]                           # the 41 quarter dates

# One case: 41 rows x 3 columns
case_id = 1
df_case = pd.DataFrame(z["Y"][case_id - 1], columns=cols, index=pd.Index(dates, name="date"))
print(df_case.head())

# The observed history: 41 rows x 3 columns
df_hist = pd.DataFrame(z["y_hist"], columns=cols, index=pd.Index(dates, name="date"))
print(df_hist.head())

# All 100 cases in one long table: 4,100 rows (100 x 41)
df_all = pd.DataFrame(z["Y"].reshape(-1, 3), columns=cols)
df_all.insert(0, "date", np.tile(dates, len(z["case_ids"])))
df_all.insert(0, "case_id", np.repeat(z["case_ids"], len(dates)))
print(df_all.head())
```

Save as CSV if you want (the 4,100-row table is the same content as `curves_quarterly.csv`, without the rates):

```python
df_case.to_csv("npz_case1.csv")              # one case, keeps the date as the first column
df_hist.to_csv("npz_history.csv")
df_all.to_csv("npz_all_cases.csv", index=False)
```

### How to open and view the `.npy` files

`grid_varying_cells.npy` holds one array of shape 100 x 42,512 x 6 (float32). The six column names are not stored in the file, so you must add them. `grid_active_cell_ids.npy` holds the row numbers (`cell_id`) of the 42,512 active cells, so it can label the rows.

```python
import numpy as np, pandas as pd
OUT = "./"

A   = np.load(OUT + "grid_varying_cells.npy", mmap_mode="r")   # mmap: reads from disk only what you use
ids = np.load(OUT + "grid_active_cell_ids.npy")                # (42512,)
names = ["porv", "poro", "permx", "permy", "tranx", "trany"]   # axis 2, in this order
print(A.shape)                                                 # (100, 42512, 6)

# One case: 42,512 rows (active cells) x 6 columns, labelled by cell_id
case_id = 1
df_grid = pd.DataFrame(A[case_id - 1], columns=names)
df_grid.insert(0, "cell_id", ids)
print(df_grid.head())
```

Save as CSV if you want. One case is about 42,000 rows. All 100 cases stacked would be about 4.25 million rows (a large file), so save one case at a time:

```python
df_grid.to_csv("npy_case1_grid.csv", index=False)
```

To join with the template (coordinates `i`, `j`, `k`, `x`, `y`, `z` and so on), merge on `cell_id`:

```python
tmpl = pd.read_csv(OUT + "grid_cells_template.csv")
df_full = df_grid.merge(tmpl, on="cell_id")
```

## 4. What the variables mean

Units follow the organiser: oil and water in STB (stock tank barrels), gas in MSCF (thousand standard cubic feet), rates per day.

### `cases.csv` (one row per case)

| Column | Meaning |
|---|---|
| `case_id` | Case number 1-100. The key for all files. |
| `item` | The original label, `Case 1` ... |
| `split` | `train` (cases 1-70), `validation` (71-85), `test` (86-100) |
| `cv_fold` | Cross-validation fold 0-4 for the 85 non-test cases. `-1` for test cases. |
| `source_file` | Workbook that holds the case's grid |
| `fault_trans` | Fault transmissibility: how easily fluid crosses a crack in the rock (0.05-0.15) |
| `poro_mult` | Porosity multiplier: scales the empty space in the rock (0.80-1.49) |
| `perm_mult` | Permeability multiplier: scales how easily fluid flows (0.52-9.78) |
| `aquifer_pv` | Aquifer pore volume: size of the water zone that pushes the oil (54-200, unit not stated) |
| `*_s` (four columns) | The same four parameters scaled to about 0-1 with the train bounds |

### `param_bounds.csv`
`parameter`, `train_min`, `train_max`: smallest and largest value among the 70 train cases.

### `history_observed.csv` (41 quarters, 1998-2008)

| Column | Meaning |
|---|---|
| `date`, `t_days` | Quarter start date; days since 1998-01-01 |
| `gas_cum`, `oil_cum`, `water_cum` | Total produced since the start (MSCF, STB, STB) |
| `gas_rate`, `oil_rate`, `water_rate` | Reported production rate (MSCF/d, STB/d, STB/d) |
| `*_rate_avg` (three columns) | Average rate over the quarter ending on `date`. Comparable with the cases. Empty in the first row. |
| `water_inj_rate` | Water injected into the reservoir (STB/d). Only the history has it. |

### `curves_native.csv` and `curves_quarterly.csv` (curves of the 100 cases)

| Column | Meaning |
|---|---|
| `case_id` | Case number |
| `split` | Train, validation or test (only in `curves_quarterly.csv`) |
| `date`, `t_days` | Time stamp; days since 1998-01-01 |
| `gas_cum`, `oil_cum`, `water_cum` | Total produced since the start |
| `gas_rate`, `oil_rate`, `water_rate` | Reported rate |
| `*_rate_avg` | Average rate over the quarter ending on `date` (only in `curves_quarterly.csv`) |

`curves_native.csv` has 156-197 time steps per case. `curves_quarterly.csv` has the same 41 dates for every case, and the same dates as the history.

### `forecast_dates.csv`

| Column | Meaning |
|---|---|
| `step` | 1-80 |
| `date` | Quarter start date, 2008-04-01 to 2028-01-01 |
| `t_days` | Days since 1998-01-01 (same axis as the curves) |
| `years_after_history` | Years since 2008-01-01 (0.25-20) |

The forecast template to fill is `09 Template Deliverable.xlsx` (included, unchanged). It has 10 tabs, `Case 1` to `Case 10`. Each tab has the columns `Date`, `Gas production cumulative [MSCF]`, `Oil production cumulative [STB]`, `Water production cumulative [STB]`. Two things to know:
- The dates in the template are text in `MM/DD/YYYY` form. Keep that form when you fill it.
- Tab `Case 1` has only 40 rows (to 01/01/2018). The other nine tabs have 80 rows, which match `forecast_dates.csv`.

### `grid_cells_template.csv` (one row per grid cell)

The grid is a 3D box of 53 x 55 x 31 cells. Only 42,512 cells (47%) are active. These columns are the same in every case.

| Column | Meaning |
|---|---|
| `cell_id`, `active` | Row number; `True` if the cell is part of the reservoir |
| `i`, `j`, `k` | Cell address |
| `x`, `y`, `z` | Map position and depth of the cell centre |
| `permz`, `tranz` | Vertical permeability; vertical flow capacity between neighbour cells |
| `ntg` | Share of the cell that is reservoir rock (always 1) |
| `fipnum`, `pvtnum` | Region numbers (1 or 2) |
| `swl`, `swcr`, `swu` | Water saturation limits (share of pore space filled with water) |
| `sgl`, `sgcr`, `sowcr` | Gas and residual-oil saturation limits |
| `pressure` | Pressure at the start |
| `swat`, `soil`, `sgas` | Water, oil and gas share of the pore space at the start |
| `rs`, `rv` | Gas dissolved in oil (0.3633); oil in gas (0) |

Cells that are not active have empty values.

Note: this file is mainly useful for spatial models, because it gives the position and the fixed properties of each cell. For a model that uses only the 4 parameters, you can skip it (see the remark in section 1).

### `grid_varying_cells.npy` (the part of the grid that changes)

Axis 0 = case (`case_id` - 1). Axis 1 = active cell. Axis 2 = column: `porv` (pore volume), `poro` (porosity: share of rock that is empty space), `permx` and `permy` (permeability, ease of flow), `tranx` and `trany` (flow capacity to the neighbour cell).

### `grid_case_features.csv` (one row per case)

| Column | Meaning |
|---|---|
| `total_pv` | Total pore volume of the grid |
| `aquifer_pv_grid` | Pore volume of the aquifer cells |
| `reservoir_pv` | Pore volume outside the aquifer |
| `mean_poro` | Average porosity |
| `mean_permx`, `mean_permx_zone` | Average permeability, over all cells and over the low-permeability zone |
| `mean_tranx`, `mean_trany` | Average flow capacity to the neighbour cell |

These are combinations of the 4 parameters. They add little to a model that already has them.

### `grid_column_catalog.csv` and `grid_case_checks.csv`
- Catalog: for each of the 29 raw grid columns, its short name, original header, group, and whether it changes between cases (only 6 do).
- Checks: per case, the number of active cells (always 42,512) and whether the active cells match case 1 (always yes).

## 5. Rules that hold in every file

| Rule | Meaning |
|---|---|
| Key | `case_id` (1-100) is the same everywhere. Array index = `case_id` - 1. |
| Time axis | `t_days` = days since 1998-01-01. History and cases end on 2008-01-01 (day 3,652). |
| Missing values | Empty cells. The raw `-999.25` is gone. |
| Precision | CSV numbers are rounded to 15 significant digits (an Excel limit). The `.npy` and `.npz` files keep full precision. |

## 6. Cautions

- **No simulated data after 2008-01-01.** The forecast has no answer key.
- **Compare cumulative values, not reported rates.** The history and the cases follow different rate conventions. If you need a rate, use `*_rate_avg`.
- **Keep the test cases (86-100) locked** until the final score. `cv_fold = -1` marks them.
- **Five validation/test parameter values lie slightly outside the train range** (cases 75, 82, 87, 98; at most 3.2% of the range). The scaled values can be a little below 0 or above 1. Do not clip them.
- **Fault transmissibility is not visible in the grid.** Keep it as a direct input.
- **Do not save these CSVs from Excel.** If you open one, click **Don't Convert**. Or import it with Data > From Text/CSV.

## 7. Quick start

```python
import numpy as np, pandas as pd
OUT = "./"                               # the folder that holds these files
P = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]

cases = pd.read_csv(OUT + "cases.csv")
z = np.load(OUT + "curves_quarterly_cum.npz", allow_pickle=False)
Y, y_hist = z["Y"], z["y_hist"]          # (100, 41, 3) and (41, 3); order gas, oil, water

train = cases[cases.split == "train"]
X_train = train[P].to_numpy()
Y_train = Y[train.case_id.to_numpy() - 1]   # array index = case_id - 1
```
