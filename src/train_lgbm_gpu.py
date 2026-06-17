import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from train_gpu_best import DATA_DIR, ID_COL, OUTPUT_DIR, SEED, TARGET, build_features, print_environment

warnings.filterwarnings("ignore")


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def rmse(y_true, pred):
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def train_lgbm_gpu(x_train, x_test, y):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 2.5,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 1,
        "device_type": "gpu",
        "device": "gpu",
        "gpu_platform_id": 1,
        "gpu_device_id": 0,
        "max_bin": 255,
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(x_train), dtype=np.float32)
    pred = np.zeros(len(x_test), dtype=np.float32)
    scores = []
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
        log(f"========== LightGBM GPU fold {fold}/5 ==========")
        model = lgb.LGBMRegressor(**params)
        model.fit(
            x_train.iloc[tr_idx],
            y.iloc[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y.iloc[va_idx])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(250)],
        )
        va_pred = np.clip(model.predict(x_train.iloc[va_idx], num_iteration=model.best_iteration_), 0, 1)
        te_pred = np.clip(model.predict(x_test, num_iteration=model.best_iteration_), 0, 1)
        score = rmse(y.iloc[va_idx], va_pred)
        log(f"fold {fold} RMSE={score:.6f}, best_iter={model.best_iteration_}")
        oof[va_idx] = va_pred
        pred += te_pred / kf.n_splits
        scores.append(float(score))
        best_iters.append(int(model.best_iteration_))
    oof_score = rmse(y, oof)
    log(f"OOF RMSE={oof_score:.6f}; mean={np.mean(scores):.6f}; std={np.std(scores):.6f}")
    return oof, pred, oof_score, scores, best_iters, params


def main():
    start = time.perf_counter()
    print_environment()
    log(f"LightGBM version: {lgb.__version__}")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    x_train, x_test, y, feature_cols = build_features(train, test)
    oof, pred, oof_score, scores, best_iters, params = train_lgbm_gpu(x_train, x_test, y)

    pd.DataFrame({ID_COL: train[ID_COL], TARGET: oof}).to_csv(OUTPUT_DIR / "oof_lgbm_gpu_fe.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}).to_csv(OUTPUT_DIR / "pred_lgbm_gpu_fe.csv", index=False)
    sub = sample.copy()
    sub[TARGET] = pred
    sub.to_csv(OUTPUT_DIR / "submission_lgbm_gpu_fe.csv", index=False)
    summary = {
        "model": "LightGBM GPU",
        "features": "domain interactions + count encoding + strict OOF target encoding",
        "oof_rmse": float(oof_score),
        "fold_rmse": scores,
        "best_iterations": best_iters,
        "params": params,
        "n_features": len(feature_cols),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (OUTPUT_DIR / "lgbm_gpu_fe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("saved outputs/submission_lgbm_gpu_fe.csv")
    print(sub[TARGET].describe(), flush=True)


if __name__ == "__main__":
    main()
