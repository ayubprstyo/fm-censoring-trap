# Do time-series foundation models inherit the demand-censoring trap?

Code and result files for the paper

> Prasetyo, A., Amalia, F., "Do Time-Series Foundation Models Inherit the Demand-Censoring Trap? A closed-loop newsvendor study in fresh retail," submitted to the *International Journal of Industrial Engineering & Production Research* (IJIEPR), 2026.

When an item sells out, recorded sales understate demand. A replenishment policy that learns from its own sales can therefore drift into chronic under-ordering. This repository places three zero-shot time-series foundation models (Chronos-2, TimesFM-2.5, TiRex) inside a closed-loop daily newsvendor on the [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) fresh-retail data and compares them with an empirical-quantile policy and a Kaplan-Meier policy.

## Main results (hourly-based ground truth, 1,000 series, 60 simulated days)

| | β = 0.5 | β = 0.8 | β = 0.9 |
|---|---|---|---|
| Foundation models given true demand vs. empirical oracle | −1.3 to −2.3% | −3.9 to −4.9% | −3.8 to −4.0% |
| Foundation models learning from own sales vs. own oracle | +38 to +47% | +12 to +15% | +7 to +14% |
| Foundation models (naive) vs. empirical policy (naive) | +21 to +29% | −10 to −13% | −21 to −26% |
| Chronos-2 with stockout covariate vs. Kaplan-Meier | +28.5% | +1.0% | −11.4% |
| Tobit (censored normal, 28 d) vs. empirical oracle | +1.2% | +1.3% | +5.6% |
| Chronos-2 with stockout covariate vs. Tobit | +28.4% | +5.7% | −2.9% |
| Foundation models (naive) vs. empirical (naive), both 28-day context | +19 to +36% | −9 to −13% | −21 to −27% |

All intervals (paired bootstrap, 2,000 resamples) exclude zero; see `results/numbers.json` and `results/nb05/nb05_numbers.json`.

The robustness run (notebook 05) shows that the reversal across critical ratios is not caused by the longer context of the foundation models, and that a Tobit policy is a much stronger classical benchmark than Kaplan-Meier at high critical ratios.

## Repository layout

```
notebooks/
  01_vn2_censoring_check.py          VN2 weekly data, empirical policies only (CPU, < 1 min)
  02_pilot_freshretailnet.py         go/no-go pilot on FreshRetailNet-50K, empirical policies (CPU)
  03_fm_loop_median_groundtruth.py   first full run with foundation models, median-filled ground truth (GPU)
  04_fm_loop_hourly_groundtruth.py   PRIMARY ANALYSIS: hourly-based ground truth, trajectories, NaN diagnostic (GPU)
  05_robustness_context_tobit.py     robustness: equal 28-day vs full-history context for all forecasters,
                                     and a Tobit (censored-normal, EM) baseline (GPU)
  *.ipynb                            the same notebooks in Jupyter format (converted with jupytext)
analysis/
  compute_numbers.py                 paired contrasts and CIs for Tables 3, 4 and 6 -> results/numbers.json
  compute_nb05.py                    robustness contrasts for Table 7 -> results/nb05/nb05_numbers.json
  make_figures.py                    Figures 1-3 (300 dpi) and Figures_data_Excel.xlsx
results/
  nb03/                              output of notebook 04 (primary analysis)
  nb02/                              output of notebook 03 (superseded, kept for transparency)
  nb05/                              output of notebook 05 (robustness, Table 7)
  numbers.json                       numbers used in the paper
  vn2_check_output.txt               output of notebook 01
figures/                             Fig1-Fig3 and the Excel workbook
```

## Reproducing the results

**Notebooks 03, 04 and 05** were run on Kaggle (GPU T4 x1, Internet on). Create a notebook, paste the script, and select *Run All*. Notebook 04 installs its own packages and takes about 60-90 minutes; outputs are written to `/kaggle/working`. Both notebooks have a mock mode for a quick local test without data or models: `NB03_MOCK=1 python notebooks/04_fm_loop_hourly_groundtruth.py` (`NB02_MOCK=1` for notebook 03, `NB05_MOCK=1` for notebook 05).

**Notebook 02** runs on CPU in a few minutes with internet access to Hugging Face.

**Notebook 01** needs the VN2 data:

```bash
git clone --depth 1 https://github.com/SupChains/VN2.git
python notebooks/01_vn2_censoring_check.py --vn2 VN2/Data
```

**Tables and figures** from the saved results, without a GPU:

```bash
pip install -r requirements.txt
python analysis/compute_numbers.py --results results/nb03 --out results/numbers.json
python analysis/make_figures.py --results results/nb03 --out figures
python analysis/compute_nb05.py results
```

## Versions used

| Component | Version / revision |
|---|---|
| chronos-forecasting | 2.3.2 |
| timesfm | 3.0.2 |
| tirex-ts | 1.4.2 |
| amazon/chronos-2 | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| google/timesfm-2.5-200m-pytorch | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| NX-AI/TiRex | `43ebaaff068c1390785c387b3e5178df3f4c92ca` |
| Dingdong-Inc/FreshRetailNet-50K | `c374e85975c82816c495acfa904cc55ab14a47d3` (data/train.parquet) |

The notebooks load the latest model and data revisions. The revisions above are the ones that produced the saved results (`results/nb03/model_and_data_revisions.json`). To reproduce them exactly, pin these revisions when loading.

## Experimental set-up in brief

- Ground truth: 1,000 series with complete 90-day histories and at most 20% stockout days (06:00-22:00). Demand on stockout days = in-stock-hour sales scaled by the series' intraday profile; clean-day median when in-stock hours cover < 20% of normal sales (45.5% of reconstructed days). Median and 80th-percentile fills serve as sensitivity checks.
- Loop: daily perishable newsvendor, lost sales, zero lead time, c_u = 1, critical ratio β ∈ {0.5, 0.8, 0.9}; days 1-30 warm-up, days 31-90 simulated.
- Variants: *oracle* (true demand), *naive* (own sales), *masked* (sold-out days as missing), *covariate* (sold-out flag as past covariate, Chronos-2 only).

## Data

- FreshRetailNet-50K: see the dataset card on Hugging Face for terms of use. The raw data are not redistributed here.
- VN2: SupChains, CC BY 4.0 (https://github.com/SupChains/VN2).

## Use of AI tools

An AI assistant (Claude, Anthropic) helped search the literature, write and run the simulation and analysis code, and draft the paper. The authors designed the study, checked the code, numbers and references, and take responsibility for the content.

## License

Code: MIT (see `LICENSE`). Result files and figures: CC BY 4.0.

## Citation

See `CITATION.cff`.
