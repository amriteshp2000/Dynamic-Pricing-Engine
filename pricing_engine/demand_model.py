import time
import os
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Metrics
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.special import expit
from sklearn.model_selection import train_test_split

# LightGBM
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
import tensorflow_lattice as tfl
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM

# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelConfig:
    price_col: str = "log_price"
    group_col: Optional[str] = None
    categorical_cols: List[str] = field(default_factory=list)
    
    def validate(self, df: pd.DataFrame, features: List[str]):
        if self.price_col not in features:
            raise ValueError(f"Price col '{self.price_col}' missing from features")

# ============================================================
# Base Interface (With Serialization)
# ============================================================

class DemandModel(ABC):
    def __init__(self, name: str, role: str, prediction_type: str, 
                 config: Optional[ModelConfig] = None,
                 hyperparams: Optional[Dict[str, Any]] = None):
        self.name = name
        self.role = role
        self.prediction_type = prediction_type
        self.config = config or ModelConfig()
        self.hyperparams = hyperparams or {}
        self._fitted = False
        self._features_used = []

    def _split_val(self, df, target, val_ratio=0.2):
        time_cols = [c for c in df.columns if 'date' in c.lower() or 'week' in c.lower()]
        if time_cols:
            sort_col = time_cols[0] 
            df_sorted = df.sort_values(sort_col)
            cutoff = int(len(df) * (1 - val_ratio))
            return df_sorted.iloc[:cutoff], df_sorted.iloc[cutoff:]
        return train_test_split(df, test_size=val_ratio, random_state=42)

    @abstractmethod
    def fit(self, df: pd.DataFrame, features: List[str], target: str): pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray: pass

    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        if not self._fitted: raise RuntimeError(f"{self.name} not fitted")
        if self.prediction_type == "expectation":
            return {"rmse": np.sqrt(mean_squared_error(df[target], self.predict(df)))}
        return {
            "logloss": log_loss(df[target], self.predict(df)),
            "auc": roc_auc_score(df[target], self.predict(df))
        }

    def _validate_prediction_input(self, df: pd.DataFrame):
        if not self._fitted: raise RuntimeError(f"{self.name} not fitted")
        missing = set(self._features_used) - set(df.columns)
        if missing and not all(m in df.index.names for m in missing):
            raise ValueError(f"Missing features: {missing}")

    # --- Standard Saving (For non-Keras models) ---
    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)

# ============================================================
# Keras Model Mixin (Handles Special Saving)
# ============================================================
class KerasDemandModel(DemandModel):
    """Mixin to handle safe saving of Keras objects"""
    def save(self, base_path: str):
        # 1. Create a directory for this model
        model_dir = base_path + "_artifacts"
        os.makedirs(model_dir, exist_ok=True)
        
        # 2. Save Keras model natively
        keras_path = os.path.join(model_dir, "keras_model")
        self.model.save(keras_path)
        
        # 3. Temporarily detach model to pickle the wrapper
        keras_ref = self.model
        self.model = None
        
        # 4. Save wrapper logic
        wrapper_path = os.path.join(model_dir, "wrapper.pkl")
        joblib.dump(self, wrapper_path)
        
        # 5. Restore reference
        self.model = keras_ref
        
        # 6. Save a pointer file
        with open(base_path, 'w') as f:
            f.write(model_dir)

    @classmethod
    def load(cls, base_path: str):
        with open(base_path, 'r') as f:
            model_dir = f.read().strip()
        wrapper = joblib.load(os.path.join(model_dir, "wrapper.pkl"))
        keras_path = os.path.join(model_dir, "keras_model")
        wrapper.model = tf.keras.models.load_model(keras_path)
        return wrapper

# ============================================================
# Shared Utilities
# ============================================================

def _safe_clip(pred: np.ndarray) -> np.ndarray:
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(pred, 1e-5, 1.0 - 1e-5)

# ============================================================
# Model 1: LGBM (Production)
# ============================================================
class LGBMTweedie(DemandModel):
    def __init__(self, config: ModelConfig = None, hyperparams: Dict = None):
        super().__init__("LGBM_Tweedie", "production", "probability", config, hyperparams)
        self.model = None
        self.cat_cols = []

    def fit(self, df, features, target):
        self.config.validate(df, features)
        self._features_used = features
        train_df, val_df = self._split_val(df, target)
        
        X_train, y_train = train_df[features].copy(), train_df[target]
        X_val, y_val = val_df[features].copy(), val_df[target]

        if self.config.categorical_cols:
            self.cat_cols = [c for c in self.config.categorical_cols if c in features]
        else:
            self.cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ('category', 'object')]
            
        for c in self.cat_cols:
            X_train[c] = X_train[c].astype('category')
            X_val[c] = X_val[c].astype('category')

        mono_constraints = [-1 if f == self.config.price_col else 0 for f in features]
        
        params = {
            "n_estimators": 5000, "learning_rate": 0.02, "num_leaves": 31,
            "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8,
            **self.hyperparams
        }

        self.model = lgb.LGBMClassifier(
            objective="binary", monotone_constraints=mono_constraints,
            monotone_constraints_method='advanced', random_state=42, n_jobs=-1, verbose=-1, **params
        )
        
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=callbacks)
        self._fitted = True

    def predict(self, df):
        self._validate_prediction_input(df)
        X = df[self._features_used].copy()
        for c in self.cat_cols: X[c] = X[c].astype('category')
        return self.model.predict_proba(X)[:, 1]

# ============================================================
# Model 2: DeepFM (Research) - Uses KerasDemandModel
# ============================================================
class DeepFMModel(KerasDemandModel):
    def __init__(self, config: ModelConfig = None, hyperparams: Dict = None):
        super().__init__("DeepFM", "research", "probability", config, hyperparams)
        self.model = None; self.feature_names = None; self.encoders = {}; self.scaler = MinMaxScaler()

    def fit(self, df, features, target):
        self.config.validate(df, features)
        self._features_used = features
        train_df, val_df = self._split_val(df, target)
        full_X = pd.concat([train_df[features], val_df[features]])
        
        if self.config.categorical_cols:
            sparse_features = [c for c in self.config.categorical_cols if c in features]
        else:
            sparse_features = [f for f in features if full_X[f].dtype.name in ("category", "object")]
        dense_features = [f for f in features if f not in sparse_features]

        X_train, X_val = train_df[features].copy(), val_df[features].copy()

        if dense_features:
            self.scaler.fit(full_X[dense_features].fillna(0))
            X_train[dense_features] = self.scaler.transform(X_train[dense_features].fillna(0))
            X_val[dense_features] = self.scaler.transform(X_val[dense_features].fillna(0))

        for feat in sparse_features:
            le = LabelEncoder().fit(full_X[feat].astype(str))
            self.encoders[feat] = le
            X_train[feat] = le.transform(X_train[feat].astype(str)) + 1
            X_val[feat] = le.transform(X_val[feat].astype(str)) + 1

        feature_columns = ([SparseFeat(f, len(self.encoders[f].classes_) + 2, self.hyperparams.get("embedding_dim", 8)) for f in sparse_features] + [DenseFeat(f, 1) for f in dense_features])
        self.feature_names = get_feature_names(feature_columns)

        self.model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, task="binary")
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

        train_inp = {f: X_train[f].values for f in self.feature_names}
        val_inp = {f: X_val[f].values for f in self.feature_names}
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)
        self.model.fit(train_inp, train_df[target].values, validation_data=(val_inp, val_df[target].values),
                       batch_size=self.hyperparams.get("batch_size", 512), epochs=self.hyperparams.get("epochs", 100), callbacks=[es], verbose=0)
        self._fitted = True

    def predict(self, df):
        self._validate_prediction_input(df)
        X = df[self._features_used].copy()
        dense_features = [f for f in self.feature_names if f not in self.encoders]
        if dense_features: X[dense_features] = self.scaler.transform(X[dense_features].fillna(0))
        for feat, le in self.encoders.items():
            X[feat] = X[feat].astype(str).map(lambda x: le.transform([x])[0] + 1 if x in le.classes_ else 0)
        return _safe_clip(self.model.predict({f: X[f].values for f in self.feature_names}, batch_size=1024).flatten())

# ============================================================
# Model 3: TF Lattice (Safety) - Uses KerasDemandModel
# ============================================================
class TFLatticeModel(KerasDemandModel):
    def __init__(self, config: ModelConfig = None, hyperparams: Dict = None):
        super().__init__("TF_Lattice", "safety", "probability", config, hyperparams)
        self.model = None

    def fit(self, df, features, target):
        self.config.validate(df, features)
        self._features_used = [f for f in features if df[f].dtype.kind in 'biufc']
        train_df, val_df = self._split_val(df, target)
        
        # Init Bias
        mean_rate = df[target].mean()
        init_bias = np.log(np.clip(mean_rate, 1e-5, 1-1e-5) / (1 - np.clip(mean_rate, 1e-5, 1-1e-5)))
        
        inputs, calibrators = [], []
        lattice_monotonicities = []
        
        for f in self._features_used:
            inp = tf.keras.layers.Input(shape=(1,), name=f)
            inputs.append(inp)
            
            clean_vals = train_df[f].dropna().values
            if len(clean_vals) > 0:
                span = clean_vals.max() - clean_vals.min()
                kp = np.linspace(clean_vals.min() - 0.05*span, clean_vals.max() + 0.05*span, 20)
            else:
                kp = np.linspace(0, 1, 20)

            if f == self.config.price_col:
                calib_mono = -1
                lattice_mono = 1
            else:
                calib_mono = 0
                lattice_mono = 0
            
            lattice_monotonicities.append(lattice_mono)
            
            cal = tfl.layers.PWLCalibration(
                input_keypoints=kp, output_min=0.0, output_max=1.0, monotonicity=calib_mono
            )(inp)
            calibrators.append(cal)

        lattice = tfl.layers.Lattice(
            lattice_sizes=[3] * len(self._features_used), 
            output_min=0.0, output_max=1.0, 
            monotonicities=lattice_monotonicities
        )(calibrators)
        
        out = tf.keras.layers.Dense(
            1, activation="sigmoid", 
            kernel_constraint=tf.keras.constraints.NonNeg(),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.1, maxval=1.0),
            bias_initializer=tf.keras.initializers.Constant(init_bias)
        )(lattice)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="binary_crossentropy", metrics=['AUC'])
        
        def prep(d): return [d[f].fillna(train_df[f].median()).values for f in self._features_used]
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(
            prep(train_df), train_df[target].values,
            validation_data=(prep(val_df), val_df[target].values),
            batch_size=1024, epochs=self.hyperparams.get("epochs", 200),
            callbacks=[es], verbose=0
        )
        self._fitted = True

    def predict(self, df):
        self._validate_prediction_input(df)
        test_x = [df[f].fillna(df[f].median()).values for f in self._features_used]
        return _safe_clip(self.model.predict(test_x, batch_size=4096).flatten())

# ============================================================
# Model 4: Hierarchical Bayes
# ============================================================
class HierarchicalBayesianLogit(DemandModel):
    def __init__(self, beta_prior: float = -0.03, config: ModelConfig = None, hyperparams: Dict = None):
        super().__init__("HierarchicalBayes", "cold_start", "probability", config, hyperparams)
        self.beta_price = beta_prior; self.mu_global = 0.0; self.alpha_map = {}

    def _get_group_series(self, df):
        if not self.config.group_col: return None
        if self.config.group_col in df.columns: return df[self.config.group_col]
        if df.index.name == self.config.group_col: return pd.Series(df.index, index=df.index)
        temp = df.reset_index()
        return temp[self.config.group_col] if self.config.group_col in temp.columns else None

    def fit(self, df, features, target):
        self.config.validate(df, features); self._features_used = features
        smoothing_K = self.hyperparams.get("smoothing_K", 10.0)
        rate = df[target].mean()
        self.mu_global = np.log(np.clip(rate, 1e-5, 1-1e-5) / (1 - np.clip(rate, 1e-5, 1-1e-5)))
        
        group_series = self._get_group_series(df)
        if self.config.group_col and group_series is not None:
            tmp = pd.DataFrame({"target": df[target].values, "price": df[self.config.price_col].values, "group": group_series.values})
            agg = tmp.groupby("group")["target"].agg(['sum', 'count'])
            smoothed_rate = (agg['sum'] + smoothing_K * rate) / (agg['count'] + smoothing_K)
            smoothed_rate = np.clip(smoothed_rate, 1e-5, 1 - 1e-5)
            implied_alpha = np.log(smoothed_rate / (1 - smoothed_rate)) - (self.beta_price * tmp.groupby("group")["price"].mean())
            self.alpha_map = implied_alpha.to_dict()
        self._fitted = True

    def predict(self, df):
        self._validate_prediction_input(df)
        group_series = self._get_group_series(df)
        alpha = group_series.map(self.alpha_map).fillna(self.mu_global) if (self.config.group_col and group_series is not None) else self.mu_global
        return _safe_clip(expit(alpha + self.beta_price * df[self.config.price_col]))

# ============================================================
# Helpers
# ============================================================
def create_monotonicity_probe(df, features, price_col, price_values):
    probe_data = {}
    for feat in features:
        if feat == price_col: probe_data[feat] = price_values
        elif df[feat].dtype.kind in 'biufc': probe_data[feat] = [df[feat].median()] * len(price_values)
        else: probe_data[feat] = [df[feat].mode()[0]] * len(price_values)
    return pd.DataFrame(probe_data)