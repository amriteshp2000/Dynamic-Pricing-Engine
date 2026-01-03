import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# ============================================================
# Metrics
# ============================================================

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.special import expit

# ============================================================
# Shared Evaluation Utilities
# ============================================================

def _safe_clip(pred: np.ndarray) -> np.ndarray:
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(pred, 0.0, 1.0)

def eval_binary(y_true, y_pred):
    y_pred = _safe_clip(y_pred)
    return {
        "logloss": log_loss(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred),
    }

def eval_regression(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

# ============================================================
# Base Interface
# ============================================================

class DemandModel(ABC):
    """
    Base contract for all demand models.
    """

    def __init__(self, name: str, role: str, prediction_type: str):
        self.name = name
        self.role = role  # production | safety | research | cold_start
        self.prediction_type = prediction_type  # probability | expectation

    @abstractmethod
    def fit(self, df: pd.DataFrame, features: List[str], target: str):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        pass

    # ------------------------------
    # Optional but REQUIRED hooks
    # ------------------------------
    def calibrate(self, df: pd.DataFrame, target: str):
        """
        Hook for Platt / Isotonic calibration.
        No-op by default.
        """
        return self

    def benchmark_latency(self, df: pd.DataFrame, n_runs: int = 300) -> float:
        start = time.time()
        for _ in range(n_runs):
            _ = self.predict(df)
        return (time.time() - start) / n_runs


# ============================================================
# Model 1 — LightGBM Tweedie (Revenue King)
# ============================================================

import lightgbm as lgb

class LGBMTweedie(DemandModel):
    """
    Predicts expected value (booking count / revenue).
    """

    def __init__(self):
        super().__init__(
            name="LGBM_Tweedie",
            role="production",
            prediction_type="expectation",
        )
        self.model = None

    def fit(self, df, features, target):
        X = df[features].copy()
        for c in X.select_dtypes(include=["object"]).columns:
            X[c] = X[c].astype("category")

        self.model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.3,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, df[target])

    def predict(self, df):
        pred = self.model.predict(df)
        return np.maximum(pred, 0.0)

    def evaluate(self, df, target):
        return eval_regression(df[target], self.predict(df))


# ============================================================
# Model 2 — DeepFM (Interaction / Discovery)
# ============================================================

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
from tensorflow.keras.optimizers import Adam

class DeepFMModel(DemandModel):
    def __init__(self):
        super().__init__(
            name="DeepFM",
            role="research",
            prediction_type="probability",
        )
        self.model = None
        self.feature_names = None
        self.encoders = {}
        self.scaler = MinMaxScaler()

    def fit(self, df, features, target):
        X = df[features].copy()

        sparse_features = [
            f for f in features if df[f].dtype.name in ("category", "object")
        ]
        dense_features = [f for f in features if f not in sparse_features]

        if dense_features:
            X[dense_features] = self.scaler.fit_transform(X[dense_features])

        for feat in sparse_features:
            le = LabelEncoder()
            X[feat] = le.fit_transform(X[feat].astype(str)) + 1
            self.encoders[feat] = le

        feature_columns = (
            [SparseFeat(f, vocabulary_size=X[f].max() + 1, embedding_dim=8)
             for f in sparse_features]
            + [DenseFeat(f, 1) for f in dense_features]
        )

        self.feature_names = get_feature_names(feature_columns)

        self.model = DeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            task="binary",
        )
        self.model.compile(
            optimizer=Adam(0.001),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )

        model_input = {f: X[f].values for f in self.feature_names}
        self.model.fit(
            model_input,
            df[target].values,
            batch_size=512,
            epochs=5,
            verbose=0,
        )

    def predict(self, df):
        X = df.copy()

        dense_features = [f for f in self.feature_names if f not in self.encoders]
        if dense_features:
            X[dense_features] = self.scaler.transform(X[dense_features])

        for feat, le in self.encoders.items():
            X[feat] = X[feat].astype(str).map(
                lambda x: le.transform([x])[0] + 1 if x in le.classes_ else 0
            )

        model_input = {f: X[f].values for f in self.feature_names}
        return _safe_clip(self.model.predict(model_input, batch_size=512).flatten())

    def evaluate(self, df, target):
        return eval_binary(df[target], self.predict(df))


# ============================================================
# Model 3 — TensorFlow Lattice (Safety-Critical)
# ============================================================

import tensorflow as tf
import tensorflow_lattice as tfl

class TFLatticeModel(DemandModel):
    def __init__(self):
        super().__init__(
            name="TF_Lattice",
            role="safety",
            prediction_type="probability",
        )
        self.model = None
        self.features = None

    def fit(self, df, features, target):
        self.features = features
        price_col = "log_price" if "log_price" in features else "avg_price"

        inputs, calibrators = [], []

        for f in features:
            inp = tf.keras.layers.Input(shape=(1,), name=f)
            inputs.append(inp)

            keypoints = np.unique(np.quantile(df[f], np.linspace(0, 1, 10)))
            monotonicity = -1 if f == price_col else 0

            cal = tfl.layers.PWLCalibration(
                input_keypoints=keypoints,
                output_min=0.0,
                output_max=1.0,
                monotonicity=monotonicity,
            )(inp)
            calibrators.append(cal)

        lattice = tfl.layers.Lattice(
            lattice_sizes=[2] * len(features),
            output_min=0.0,
            output_max=1.0,
        )(calibrators)

        out = tf.keras.layers.Dense(1, activation="sigmoid")(lattice)
        self.model = tf.keras.Model(inputs=inputs, outputs=out)
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        self.model.fit(
            [df[f].values for f in features],
            df[target].values,
            batch_size=512,
            epochs=10,
            verbose=0,
        )

    def predict(self, df):
        return _safe_clip(
            self.model.predict([df[f].values for f in self.features]).flatten()
        )

    def evaluate(self, df, target):
        return eval_binary(df[target], self.predict(df))


# ============================================================
# Model 4 — Hierarchical Bayesian Logistic (Cold Start)
# ============================================================

class HierarchicalBayesianLogit(DemandModel):
    def __init__(self, beta_prior: float = -0.03):
        super().__init__(
            name="HierarchicalBayes",
            role="cold_start",
            prediction_type="probability",
        )
        self.beta_price = beta_prior
        self.mu_global = 0.0
        self.alpha_map = {}

    def fit(self, df, features, target):
        rate = df[target].mean()
        self.mu_global = np.log(rate / (1 - rate))

        price_effect = self.beta_price * df["log_price"]
        smoothed_y = (df[target] * len(df) + 0.5) / (len(df) + 1.0)
        implied_alpha = np.log(smoothed_y / (1 - smoothed_y)) - price_effect

        self.alpha_map = implied_alpha.groupby(df["listing_id"]).mean().to_dict()

    def predict(self, df):
        alpha = df["listing_id"].map(self.alpha_map).fillna(self.mu_global)
        logits = alpha + self.beta_price * df["log_price"]
        return _safe_clip(expit(logits))

    def evaluate(self, df, target):
        return eval_binary(df[target], self.predict(df))


# ============================================================
# Routing Logic (Single Source of Truth)
# ============================================================

def select_model(context: Dict[str, Any]) -> str:
    """
    Routing policy for production.
    """
    if context.get("cold_start", False):
        return "HierarchicalBayes"
    if context.get("price_change_large", False):
        return "TF_Lattice"
    return "LGBM_Tweedie"
