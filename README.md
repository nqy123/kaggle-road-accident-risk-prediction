# Kaggle Predicting Road Accident Risk

This project is a GPU tree-ensemble solution for Kaggle Playground Series S5E10,
`playground-series-s5e10`.

## Best Result

| File | Public RMSE | Private RMSE |
| --- | ---: | ---: |
| `outputs/submission_cat80_lgbm10_xgb10.csv` | 0.05563 | 0.05588 |

The downloaded public leaderboard snapshot puts this around the top 46%.
The target top 20% threshold is around `0.05554`, so this project is close but not yet at the requested threshold.

## Method

The best version uses interpretable road-risk feature engineering:

- Speed x curvature exposure.
- Speed per lane and curvature per lane.
- Accident count interactions with speed and curvature.
- Bad-weather and low-light indicators.
- School season, holiday, and time-of-day interactions.
- Road type, weather, lighting, speed, curvature-bin, and accident-bin interaction features.
- Count encoding and strict OOF target encoding for XGBoost/LightGBM.
- Native categorical handling for CatBoost.

Final submission:

- 80% CatBoost GPU
- 10% XGBoost GPU
- 10% LightGBM GPU

The official original dataset was tested, but it worsened OOF due to distribution shift, so it is not used in the final submission.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the main candidates:

```bash
python src/train_cat_gpu.py
python src/train_lgbm_gpu.py
$env:N_ESTIMATORS='1400'; $env:OUTPUT_SUFFIX='_short'; python src/train_gpu_best.py
```

Then recreate the final blend from saved predictions.

## Outputs

Important files:

- `outputs/submission_cat80_lgbm10_xgb10.csv`
- `outputs/oof_cat_gpu_fe.csv`
- `outputs/pred_cat_gpu_fe.csv`
- `outputs/oof_lgbm_gpu_fe.csv`
- `outputs/pred_lgbm_gpu_fe.csv`
- `outputs/oof_xgb_gpu_fe_short.csv`
- `outputs/pred_xgb_gpu_fe_short.csv`
- `outputs/experiment_log.csv`
- `outputs/best_result_summary.csv`

Raw Kaggle data is not committed.
