import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "playground-series-s5e10"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ID_COL = "id"
TARGET = "accident_risk"
SEED = 42
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "6000"))
OUTPUT_SUFFIX = os.getenv("OUTPUT_SUFFIX", "")


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        log(f"START: {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        log(f"END: {self.name}, elapsed={time.perf_counter() - self.start:.2f}s")


def print_environment():
    log("========== Environment ==========")
    log(f"Python: {sys.version.replace(chr(10), ' ')}")
    log(f"Python executable: {sys.executable}")
    log(f"Platform: {platform.platform()}")
    try:
        import xgboost as xgb
        log(f"XGBoost version: {xgb.__version__}")
    except Exception as exc:
        log(f"XGBoost version check failed: {exc}")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    log("nvidia-smi stdout:")
    print(result.stdout, flush=True)
    if result.stderr:
        log("nvidia-smi stderr:")
        print(result.stderr, flush=True)


def rmse(y_true, pred):
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def add_domain_features(df):
    """道路风险可解释特征：速度、曲率、车道、天气、照明、事故历史的交互。"""
    out = df.copy()
    out["is_bad_weather"] = out["weather"].isin(["rainy", "foggy"]).astype(int)
    out["is_low_light"] = out["lighting"].isin(["dim", "night"]).astype(int)
    out["is_rush_like"] = out["time_of_day"].isin(["morning", "evening"]).astype(int)
    out["no_road_signs"] = (~out["road_signs_present"]).astype(int)
    out["is_public_road"] = out["public_road"].astype(int)
    out["is_holiday"] = out["holiday"].astype(int)
    out["is_school"] = out["school_season"].astype(int)

    out["speed_x_curvature"] = out["speed_limit"] * out["curvature"]
    out["speed_per_lane"] = out["speed_limit"] / out["num_lanes"]
    out["curvature_per_lane"] = out["curvature"] / out["num_lanes"]
    out["accidents_per_lane"] = out["num_reported_accidents"] / out["num_lanes"]
    out["accidents_x_speed"] = out["num_reported_accidents"] * out["speed_limit"]
    out["accidents_x_curvature"] = out["num_reported_accidents"] * out["curvature"]
    out["visibility_risk"] = out["is_bad_weather"] + out["is_low_light"]
    out["complexity_score"] = (
        out["curvature"] * 2.0
        + out["speed_limit"] / 70.0
        + out["num_reported_accidents"] / 7.0
        + out["visibility_risk"]
        + out["no_road_signs"]
    )
    out["protected_context"] = out["road_signs_present"].astype(int) + out["is_public_road"]
    out["school_holiday"] = out["is_school"] * out["is_holiday"]
    out["weather_light_risk"] = out["is_bad_weather"] * out["is_low_light"]
    out["speed_bin"] = out["speed_limit"].astype(str)
    out["curvature_bin"] = pd.cut(out["curvature"], bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=False).astype(str)
    out["accident_bin"] = out["num_reported_accidents"].astype(str)

    interaction_cols = [
        ["road_type", "weather"],
        ["road_type", "lighting"],
        ["road_type", "time_of_day"],
        ["road_type", "speed_bin"],
        ["road_type", "accident_bin"],
        ["weather", "lighting"],
        ["weather", "time_of_day"],
        ["lighting", "time_of_day"],
        ["holiday", "school_season", "time_of_day"],
        ["road_signs_present", "public_road"],
        ["road_type", "weather", "lighting"],
        ["road_type", "curvature_bin", "speed_bin"],
        ["weather", "lighting", "accident_bin"],
    ]
    for cols in interaction_cols:
        name = "__".join(cols)
        out[name] = out[cols].astype(str).agg("_".join, axis=1)
    return out


def add_count_and_codes(train_x, test_x, cat_cols):
    """train+test 统一做类别编码和频次编码，不使用目标。"""
    tr = train_x.copy()
    te = test_x.copy()
    all_df = pd.concat([tr[cat_cols], te[cat_cols]], axis=0, ignore_index=True)
    for col in cat_cols:
        values = all_df[col].astype(str)
        codes, uniques = pd.factorize(values, sort=True)
        tr[f"{col}_code"] = codes[: len(tr)].astype("int32")
        te[f"{col}_code"] = codes[len(tr):].astype("int32")
        vc = pd.Series(values).value_counts()
        tr[f"{col}_count"] = tr[col].astype(str).map(vc).astype("float32")
        te[f"{col}_count"] = te[col].astype(str).map(vc).astype("float32")
    return tr, te


def add_oof_target_encoding(train_x, test_x, y, cat_cols, n_splits=5, smoothing=20):
    """严格 OOF target encoding，用于低基数类别和交互特征。"""
    tr = train_x.copy()
    te = test_x.copy()
    global_mean = float(y.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for col in cat_cols:
        log(f"target encoding: {col}")
        tr[f"{col}_te"] = global_mean
        for fold, (fit_idx, val_idx) in enumerate(kf.split(tr), 1):
            stats = pd.DataFrame({col: tr.iloc[fit_idx][col].astype(str), TARGET: y.iloc[fit_idx]}).groupby(col)[TARGET].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            tr.iloc[val_idx, tr.columns.get_loc(f"{col}_te")] = tr.iloc[val_idx][col].astype(str).map(smooth).fillna(global_mean).values
        stats_full = pd.DataFrame({col: tr[col].astype(str), TARGET: y}).groupby(col)[TARGET].agg(["mean", "count"])
        smooth_full = (stats_full["mean"] * stats_full["count"] + global_mean * smoothing) / (stats_full["count"] + smoothing)
        te[f"{col}_te"] = te[col].astype(str).map(smooth_full).fillna(global_mean).values
    return tr, te


def build_features(train, test):
    with Timer("feature construction"):
        y = train[TARGET].copy()
        tr = add_domain_features(train.drop(columns=[TARGET]))
        te = add_domain_features(test)
        cat_cols = tr.select_dtypes(include=["object", "bool"]).columns.tolist()
        tr, te = add_count_and_codes(tr, te, cat_cols)
        tr, te = add_oof_target_encoding(tr, te, y, cat_cols)
        drop_cols = [ID_COL] + cat_cols
        feature_cols = [c for c in tr.columns if c not in drop_cols]
        tr = tr[feature_cols].astype("float32")
        te = te[feature_cols].astype("float32")
        log(f"features: train={tr.shape}, test={te.shape}")
        return tr, te, y, feature_cols


def train_xgb_gpu(x_train, x_test, y):
    params = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.015,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "n_jobs": -1,
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(x_train), dtype=np.float32)
    pred = np.zeros(len(x_test), dtype=np.float32)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
        log(f"========== XGBoost GPU fold {fold}/5 ==========")
        with Timer(f"xgb fold {fold}"):
            model = XGBRegressor(**params)
            model.fit(
                x_train.iloc[tr_idx],
                y.iloc[tr_idx],
                eval_set=[(x_train.iloc[va_idx], y.iloc[va_idx])],
                verbose=250,
            )
        va_pred = np.clip(model.predict(x_train.iloc[va_idx]), 0, 1)
        te_pred = np.clip(model.predict(x_test), 0, 1)
        score = rmse(y.iloc[va_idx], va_pred)
        log(f"fold {fold} RMSE={score:.6f}")
        oof[va_idx] = va_pred
        pred += te_pred / kf.n_splits
        scores.append(float(score))
    oof_score = rmse(y, oof)
    log(f"OOF RMSE={oof_score:.6f}; mean={np.mean(scores):.6f}; std={np.std(scores):.6f}")
    return oof, pred, oof_score, scores, params


def main():
    start = time.perf_counter()
    print_environment()
    with Timer("read data"):
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    x_train, x_test, y, feature_cols = build_features(train, test)
    with Timer("GPU training"):
        oof, pred, oof_score, scores, params = train_xgb_gpu(x_train, x_test, y)

    oof_path = OUTPUT_DIR / f"oof_xgb_gpu_fe{OUTPUT_SUFFIX}.csv"
    pred_path = OUTPUT_DIR / f"pred_xgb_gpu_fe{OUTPUT_SUFFIX}.csv"
    sub_path = OUTPUT_DIR / f"submission_xgb_gpu_fe{OUTPUT_SUFFIX}.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: oof}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}).to_csv(pred_path, index=False)
    sub = sample.copy()
    sub[TARGET] = pred
    sub.to_csv(sub_path, index=False)

    summary = {
        "model": "XGBoost GPU",
        "features": "domain interactions + count encoding + strict OOF target encoding",
        "oof_rmse": float(oof_score),
        "fold_rmse": scores,
        "params": params,
        "n_features": len(feature_cols),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (OUTPUT_DIR / f"xgb_gpu_fe{OUTPUT_SUFFIX}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {sub_path}")
    print(sub[TARGET].describe(), flush=True)
    log(f"TOTAL elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
