# Team What: ExxonMobil DataWorks Challenge 2026

**Topic:** Machine learning-assisted history matching and production forecasting.

**Goal:** Build a fast ML model of reservoir simulations, match it to the field's observed production history (1998-2008), and forecast production up to 2028.

## Notes for Team What

1. **Before working, copy and paste the original datasets from the organiser (the nine Excel files, `01` to `09`) into the folder `dataset_ori/`.** The files are not in the repo (they are git-ignored), so every member must add them locally.
2. Run `notebook1_data_eng.ipynb` from the repo root if you want to rebuild `data_after_engineering/`. The first run takes about 15 minutes because it reads the 100 grid sheets. If you only need the cleaned data, use `data_after_engineering/` as it is.

## Files and folders

| Path | What it is |
|---|---|
| `dataset_ori/` | The organiser's original Excel files. Empty in the repo; you add the files locally (see above). |
| `data_after_engineering/` | Output of the data engineering notebook: cleaned CSV and NumPy files, plus `MANIFEST.csv` (file list) and `data_desc_engineered.md` (handoff guide that explains every file and column). Start with the guide. |
| `instruction/` | The competition brief: `instruction1.md` (files and submission requirements) and `instruction2_briefing_slides.md` (briefing slides). |
| `notebook1_data_eng.ipynb` | Data engineering pipeline. Reads `dataset_ori/` and writes `data_after_engineering/`. |
| `grid_loader.py` | Helper module used by notebook 1 to read the grid sheets in parallel. Keep it next to the notebook. |
| `notebook1_sample_d_cleaning.ipynb`, `notebook2_sample_model1.ipynb`, `notebook3_sample_hist_matching.ipynb` | Empty sample notebooks, placeholders for the next steps (cleaning, modelling, history matching). Feel free to edit or remove later. |
| `.gitignore` | Ignores the files in `dataset_ori/` but keeps the folder. |
