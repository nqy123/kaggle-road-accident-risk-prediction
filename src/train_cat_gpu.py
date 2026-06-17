import json
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from train_gpu_best import DATA_DIR, ID_COL, OUTPUT_DIR, SEED, TARGET, add_domain_features, print_environment

warnings.filterwarnings("ignore")


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def rmse(y_true, pred):
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def build_cat_features(train, test):
    y = train[TARGET].copy()
    tr = add_domain_features(train.drop(columns=[TARGET]))
    te = add_domain_features(test)
    drop_cols = [ID_COL]
    tr = tr.drop(columns=drop_cols)
    te = te.drop(columns=drop_cols)
    cat_cols = tr.select_dtypes(include=["object", "bool"]).columns.tolist()
    for col in cat_cols:
        tr[col] = tr[col].astype(str)
        te[col] = te[col].astype(str)
    return tr, te, y, cat_cols


def main():
    start = time.perf_counter()
    print_environment()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    x_train, x_test, y, cat_cols = build_cat_features(train, test)
    cat_idx = [x_train.columns.get_loc(c) for c in cat_cols]
    log(f"features: train={x_train.shape}, test={x_test.shape}, cat_features={len(cat_idx)}")

    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3500,
        "learning_rate": 0.035,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_seed": SEED,
        "task_type": "GPU",
        "devices": "0",
        "verbose": 250,
        "allow_writing_files": False,
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(x_train), dtype=np.float32)
    pred = np.zeros(len(x_test), dtype=np.float32)
    scores = []
    best_iters = []
    test_pool = Pool(x_test, cat_features=cat_idx)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
        log(f"========== CatBoost GPU fold {fold}/5 ==========")
        train_pool = Pool(x_train.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        valid_pool = Pool(x_train.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=300)
        va_pred = np.clip(model.predict(valid_pool), 0, 1)
        te_pred = np.clip(model.predict(test_pool), 0, 1)
        score = rmse(y.iloc[va_idx], va_pred)
        log(f"fold {fold} RMSE={score:.6f}, best_iter={model.best_iteration_}")
        oof[va_idx] = va_pred
        pred += te_pred / kf.n_splits
        scores.append(float(score))
        best_iters.append(int(model.best_iteration_))
    oof_score = rmse(y, oof)
    log(f"OOF RMSE={oof_score:.6f}; mean={np.mean(scores):.6f}; std={np.std(scores):.6f}")
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: oof}).to_csv(OUTPUT_DIR / "oof_cat_gpu_fe.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}).to_csv(OUTPUT_DIR / "pred_cat_gpu_fe.csv", index=False)
    sub = sample.copy()
    sub[TARGET] = pred
    sub.to_csv(OUTPUT_DIR / "submission_cat_gpu_fe.csv", index=False)
    summary = {
        "model": "CatBoost GPU",
        "features": "domain interactions with native categorical handling",
        "oof_rmse": float(oof_score),
        "fold_rmse": scores,
        "best_iterations": best_iters,
        "params": params,
        "cat_features": cat_cols,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (OUTPUT_DIR / "cat_gpu_fe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("saved outputs/submission_cat_gpu_fe.csv")
    print(sub[TARGET].describe(), flush=True)


if __name__ == "__main__":
    main()
