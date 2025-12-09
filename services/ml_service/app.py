from pathlib import Path
from typing import List, Optional
import json
import logging

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from joblib import dump, load
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from softsensor_sdvae import SDVAE, MUDVAE, SoftSensor


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    return {}


settings = load_config()

LOG_PATH = Path(settings.get("log_path", "/app/logs/ml_service.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("ml_service")

#Простая модель по (x, y)
MODEL_PATH = Path(settings.get("model_path", "/data/model.joblib"))
TRAIN_FILE_PATH = Path(settings.get("train_data_path", "/data/storage.csv"))

#Soft sensor по реальному датасету
SOFT_MODEL_PATH = Path(settings.get("softsensor_model_path", "/data/softsensor_model.joblib"))
SOFT_TRAIN_FILE_PATH = Path(settings.get("softsensor_train_data_path", "/data/DataSet1_60T.csv"))
SOFT_TARGET_COL = settings.get("softsensor_target_col", "35_21_1000.Q.KatKlAB.EBen70")
SOFT_SEP = settings.get("softsensor_sep", ";")
SOFT_META_PATH = Path(settings.get("softsensor_meta_path", "/data/DataSet1.json"))

SOFT_SD_MODEL_PATH = Path(settings.get("softsensor_sdvae_model_path", "/data/softsensor_sdvae.joblib"))


app = FastAPI(title="ML Service")

#кеши моделей
_model: Optional[LinearRegression] = None
_soft_model: Optional[dict] = None
_soft_sdvae_model: Optional[dict] = None


class DataPoint(BaseModel):
    x: float
    y: float


class PredictRequest(BaseModel):
    items: List[DataPoint]


class PredictResponse(BaseModel):
    y_pred: List[float]
    used_trained_model: bool


class TrainResponse(BaseModel):
    rows_used: int
    model_path: str


class SoftSensorTrainResponse(BaseModel):
    rows_used: int
    n_features: int
    model_path: str
    mse: float
    r2: float


class SoftSensorPredictRequest(BaseModel):
    n_last: int = 50


class SoftSensorPredictResponse(BaseModel):
    n_last: int
    y_true: List[float]
    y_pred: List[float]


class SoftSensorSdvaeTrainResponse(BaseModel):
    rows_used: int
    n_features: int
    model_path: str
    mse: float
    r2: float


class SoftSensorSdvaePredictRequest(BaseModel):
    n_last: int = 50


class SoftSensorSdvaePredictResponse(BaseModel):
    n_last: int
    y_true: List[float]
    y_pred: List[float]


def load_model_from_disk() -> Optional[LinearRegression]:
    global _model
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        try:
            loaded = load(MODEL_PATH)
            if isinstance(loaded, LinearRegression):
                _model = loaded
                return _model
        except Exception:
            return None
    return None


def load_softsensor_model_from_disk() -> Optional[dict]:
    global _soft_model
    if _soft_model is not None:
        return _soft_model

    if SOFT_MODEL_PATH.exists():
        try:
            loaded = load(SOFT_MODEL_PATH)
            if isinstance(loaded, dict) and "model" in loaded and "feature_names" in loaded:
                _soft_model = loaded
                return _soft_model
        except Exception:
            return None
    return None

def load_softsensor_sdvae_model_from_disk() -> Optional[dict]:
    global _soft_sdvae_model
    if _soft_sdvae_model is not None:
        return _soft_sdvae_model

    if SOFT_SD_MODEL_PATH.exists():
        try:
            loaded = load(SOFT_SD_MODEL_PATH)
            if isinstance(loaded, dict) and "sdvae_state_dict" in loaded and "mudvae_state_dict" in loaded:
                _soft_sdvae_model = loaded
                return _soft_sdvae_model
        except Exception:
            return None
    return None


def load_softsensor_dataset() -> pd.DataFrame:
    if not SOFT_TRAIN_FILE_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Soft sensor train data file not found: {SOFT_TRAIN_FILE_PATH}",
        )
    try:
        df = pd.read_csv(SOFT_TRAIN_FILE_PATH, sep=SOFT_SEP, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read soft sensor train file: {e}")

    if SOFT_TARGET_COL not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Target column '{SOFT_TARGET_COL}' not found in dataset",
        )

    df = df.select_dtypes(include=["number"])
    if SOFT_TARGET_COL not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Target column '{SOFT_TARGET_COL}' is not numeric or missing after numeric filtering",
        )

    df = df.dropna(subset=[SOFT_TARGET_COL])
    if len(df) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough rows for soft sensor training: {len(df)} (need >= 50)",
        )

    return df


def categorize_columns(meta: dict):
    cv_columns = []
    mv_columns = []
    dv_columns = []

    col_kinds = meta.get("ColumnKind", {})
    for key, value in col_kinds.items():
        if value == "CV":
            cv_columns.append(key)
        elif value == "MV":
            mv_columns.append(key)
        elif value == "DV":
            dv_columns.append(key)
    return cv_columns, mv_columns, dv_columns


def filter_columns(df: pd.DataFrame, columns):
    return list(set(columns).intersection(df.columns))


def is_piecewise_constant(df: pd.DataFrame, column_name: str, threshold: float = 0.1) -> bool:
    col = df[column_name]
    changes = (col != col.shift()).sum()
    change_ratio = changes / len(col)
    return change_ratio < threshold


def find_piecewise_constant_columns(df: pd.DataFrame, threshold: float = 0.1):
    result = []
    for col in df.columns:
        try:
            if is_piecewise_constant(df, col, threshold):
                result.append(col)
        except Exception:
            continue
    return result


def remove_piecewise_constant_columns(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    cols = find_piecewise_constant_columns(df, threshold)
    if cols:
        return df.drop(columns=cols)
    return df


def load_sdvae_base_dataframe() -> pd.DataFrame:
    """
    - читаем CSV
    - выкидываем столбцы dd.mm.yyyy
    - удаляем подряд идущие одинаковые значения таргета
    """
    if not SOFT_TRAIN_FILE_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Soft sensor train data file not found: {SOFT_TRAIN_FILE_PATH}",
        )
    try:
        data = pd.read_csv(SOFT_TRAIN_FILE_PATH, sep=SOFT_SEP, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read soft sensor train file: {e}")

    obj_cols = data.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        try:
            if data[col].astype(str).str.contains(r"\d{2}\.\d{2}\.\d{4}", na=False).any():
                data = data.drop(columns=[col])
        except Exception:
            continue

    if SOFT_TARGET_COL not in data.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Target column '{SOFT_TARGET_COL}' not found in dataset (after dropping date columns)",
        )

    data = data[data[SOFT_TARGET_COL].diff().ne(0)].reset_index(drop=True)
    return data


def prepare_sdvae_train_data():
    if not SOFT_META_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Soft sensor meta file not found: {SOFT_META_PATH}",
        )

    try:
        with open(SOFT_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read soft sensor meta file: {e}")

    cv_columns, mv_columns, dv_columns = categorize_columns(meta)

    data = load_sdvae_base_dataframe()

    #разбиение по типам
    cv_in_data = filter_columns(data, cv_columns)
    mv_in_data = filter_columns(data, mv_columns)
    dv_in_data = filter_columns(data, dv_columns)

    cv_data = data[cv_in_data].astype(float) if cv_in_data else pd.DataFrame(index=data.index)
    mv_data = data[mv_in_data].astype(float) if mv_in_data else pd.DataFrame(index=data.index)
    dv_data = data[dv_in_data].astype(float) if dv_in_data else pd.DataFrame(index=data.index)

    #удаляем piecewise-constant с порогом 0.5
    cv_data = remove_piecewise_constant_columns(cv_data, threshold=0.5)
    mv_data = remove_piecewise_constant_columns(mv_data, threshold=0.5)
    dv_data = remove_piecewise_constant_columns(dv_data, threshold=0.5)

    X = pd.concat([cv_data, mv_data, dv_data], axis=1)
    X = X.select_dtypes(include=["number"])
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] == 0:
        raise HTTPException(status_code=500, detail="No usable numeric features for SDVAE after preprocessing")

    y = data[[SOFT_TARGET_COL]].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )

    xy_train = pd.concat([X_train, y_train], axis=1)
    xy_test = pd.concat([X_test, y_test], axis=1)

    xy_scaler = MinMaxScaler()
    xy_train_scaled = xy_scaler.fit_transform(xy_train)
    xy_test_scaled = xy_scaler.transform(xy_test)

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(X_train)
    x_test_scaled = x_scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "xy_train_scaled": xy_train_scaled.astype("float32"),
        "xy_test_scaled": xy_test_scaled.astype("float32"),
        "x_train_scaled": x_train_scaled.astype("float32"),
        "x_test_scaled": x_test_scaled.astype("float32"),
        "xy_scaler": xy_scaler,
        "x_scaler": x_scaler,
        "feature_names": list(X.columns),
    }


def prepare_sdvae_last_data(n_last: int, feature_names: List[str]):
    """
    Для /softsensor_sdvae/predict_last:
    берём последние n_last строк, собираем X и y с теми же признаками
    """
    data = load_sdvae_base_dataframe()

    missing = [c for c in feature_names if c not in data.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing features for SDVAE prediction: {missing}",
        )

    if SOFT_TARGET_COL not in data.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Target column '{SOFT_TARGET_COL}' not found in dataset",
        )

    if n_last <= 0:
        n_last = 1
    if n_last > len(data):
        n_last = len(data)

    df_last = data.tail(n_last)

    X_last = df_last[feature_names].astype(float)
    y_last = df_last[SOFT_TARGET_COL].astype(float).to_numpy().reshape(-1, 1)

    return X_last, y_last


@app.get("/health")
def health():
    model_exists = MODEL_PATH.exists()
    model_loaded = load_model_from_disk() is not None

    soft_model_exists = SOFT_MODEL_PATH.exists()
    soft_model_loaded = load_softsensor_model_from_disk() is not None

    soft_sd_model_exists = SOFT_SD_MODEL_PATH.exists()
    soft_sd_model_loaded = load_softsensor_sdvae_model_from_disk() is not None

    logger.info(
        f"/health: simple_model_file_exists={model_exists}, simple_model_loaded={model_loaded}, "
        f"soft_model_file_exists={soft_model_exists}, soft_model_loaded={soft_model_loaded}, "
        f"soft_sd_model_file_exists={soft_sd_model_exists}, soft_sd_model_loaded={soft_sd_model_loaded}"
    )

    return {
        "status": "ok",
        "service": "ml_service",
        "simple_model_file_exists": model_exists,
        "simple_model_loaded": model_loaded,
        "softsensor_model_file_exists": soft_model_exists,
        "softsensor_model_loaded": soft_model_loaded,
        "softsensor_sdvae_model_file_exists": soft_sd_model_exists,
        "softsensor_sdvae_model_loaded": soft_sd_model_loaded,
    }




@app.post("/train_from_file", response_model=TrainResponse)
def train_from_file():
    """
    Обучаем простую линейную регрессию
    по данным из CSV (по умолчанию /data/storage.csv)
    """
    logger.info("train_from_file: started")
    if not TRAIN_FILE_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Train data file not found: {TRAIN_FILE_PATH}",
        )

    try:
        df = pd.read_csv(TRAIN_FILE_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read train file: {e}")

    if not {"x", "y"}.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="Train file must contain 'x' and 'y' columns",
        )

    df = df[["x", "y"]].dropna()
    if len(df) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough rows to train: {len(df)} (need >= 5)",
        )

    X = df[["x"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)

    global _model
    _model = model
    logger.info(
        f"train_from_file: finished, rows_used={len(df)}, model_path={MODEL_PATH}"
    )
    return TrainResponse(rows_used=int(len(df)), model_path=str(MODEL_PATH))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info(f"/predict: received {len(req.items)} items")
    model = load_model_from_disk()
    xs = np.array([[item.x] for item in req.items], dtype=float)

    if model is not None:
        preds = model.predict(xs)
        return PredictResponse(
            y_pred=[float(v) for v in preds],
            used_trained_model=True,
        )


    preds = [2.0 * item.x for item in req.items]
    logger.info(
        f"/predict: used trained model, returned={len(preds)} predictions"
    )
    return PredictResponse(
        y_pred=preds,
        used_trained_model=False,
    )



@app.post("/softsensor/train", response_model=SoftSensorTrainResponse)
def softsensor_train():
    logger.info("softsensor_train: started (RandomForest)")
    df = load_softsensor_dataset()

    y = df[SOFT_TARGET_COL].to_numpy(dtype=float)
    X = df.drop(columns=[SOFT_TARGET_COL])

    #убираем константные признаки
    X = X.loc[:, X.nunique() > 1]
    feature_names = list(X.columns)

    if len(feature_names) == 0:
        raise HTTPException(
            status_code=500,
            detail="No non-constant numeric features left for training",
        )

    X_values = X.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    model_info = {
        "model": model,
        "feature_names": feature_names,
        "target_col": SOFT_TARGET_COL,
    }

    SOFT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model_info, SOFT_MODEL_PATH)

    global _soft_model
    _soft_model = model_info

    logger.info(
        f"softsensor_train: finished, rows_used={len(df)}, "
        f"n_features={len(feature_names)}, mse={mse:.4f}, r2={r2:.4f}, "
        f"model_path={SOFT_MODEL_PATH}"
    )

    return SoftSensorTrainResponse(
        rows_used=int(len(df)),
        n_features=len(feature_names),
        model_path=str(SOFT_MODEL_PATH),
        mse=mse,
        r2=r2,
    )


@app.post("/softsensor/predict_last", response_model=SoftSensorPredictResponse)
def softsensor_predict_last(req: SoftSensorPredictRequest):
    """
    Берём последние n_last строк из датасета,
    считаем предсказания soft sensor модели и возвращаем y_true и y_pred
    """
    logger.info(f"softsensor_predict_last: n_last={req.n_last}")
    model_info = load_softsensor_model_from_disk()
    if model_info is None:
        raise HTTPException(
            status_code=400,
            detail="Soft sensor model is not trained yet",
        )

    df = load_softsensor_dataset()

    feature_names: List[str] = model_info["feature_names"]
    target_col: str = model_info.get("target_col", SOFT_TARGET_COL)

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Missing features in dataset: {missing_features}",
        )

    n_last = req.n_last if req.n_last > 0 else 1
    if n_last > len(df):
        n_last = len(df)

    df_last = df.tail(n_last)
    X_last = df_last[feature_names].to_numpy(dtype=float)
    y_true = df_last[target_col].to_numpy(dtype=float)

    model: RandomForestRegressor = model_info["model"]
    y_pred = model.predict(X_last)

    logger.info(
        f"softsensor_predict_last: returned y_true={len(y_true)}, y_pred={len(y_pred)}"
    )
    return SoftSensorPredictResponse(
        n_last=int(n_last),
        y_true=[float(v) for v in y_true],
        y_pred=[float(v) for v in y_pred],
    )


@app.post("/softsensor_sdvae/train", response_model=SoftSensorSdvaeTrainResponse)
def softsensor_sdvae_train():
    logger.info("softsensor_sdvae_train: started")
    data_dict = prepare_sdvae_train_data()

    xy_train_scaled = data_dict["xy_train_scaled"]
    xy_test_scaled = data_dict["xy_test_scaled"]
    x_train_scaled = data_dict["x_train_scaled"]
    x_test_scaled = data_dict["x_test_scaled"]
    y_test = data_dict["y_test"].to_numpy(dtype="float32").reshape(-1, 1)

    xy_scaler = data_dict["xy_scaler"]
    x_scaler = data_dict["x_scaler"]
    feature_names = data_dict["feature_names"]

    n_samples = xy_train_scaled.shape[0] + xy_test_scaled.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim_xy = xy_train_scaled.shape[1]
    input_dim_x = x_train_scaled.shape[1]

    hidden_dims = [1024, 256]
    latent_dim = 10
    dropout_rate = 0.1

    alpha_sd = 2e-3
    alpha_mu = 3e-4

    batch_size = 20
    max_epochs_sd = 10000
    max_epochs_mu = 10000
    patience_sd = 8
    patience_mu = 5

    sdvae = SDVAE(
        input_dim=input_dim_xy,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        alpha=alpha_sd,
    ).to(device)

    train_dataset_sdvae = TensorDataset(
        torch.tensor(xy_train_scaled, dtype=torch.float32),
        torch.tensor(xy_train_scaled, dtype=torch.float32),
    )
    train_loader_sdvae = DataLoader(train_dataset_sdvae, batch_size=batch_size, shuffle=False)

    optimizer_sd = optim.AdamW(sdvae.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler_sd = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sd, mode="min", factor=0.5, patience=3
    )

    best_loss_sd = float("inf")
    best_sd_state = sdvae.state_dict()
    counter_sd = 0

    for epoch in range(max_epochs_sd):
        sdvae.train()
        epoch_loss = 0.0
        for xy_batch, _ in train_loader_sdvae:
            xy_batch = xy_batch.to(device)

            optimizer_sd.zero_grad()
            recon_xy, mu, logvar = sdvae(xy_batch)
            loss = sdvae.loss_function(recon_xy, xy_batch, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sdvae.parameters(), max_norm=1.0)
            optimizer_sd.step()

            epoch_loss += loss.item() * xy_batch.size(0)

        epoch_loss /= len(train_dataset_sdvae)
        scheduler_sd.step(epoch_loss)

        if epoch_loss < best_loss_sd:
            best_loss_sd = epoch_loss
            best_sd_state = sdvae.state_dict()
            counter_sd = 0
        else:
            counter_sd += 1
            if counter_sd >= patience_sd:
                break

    sdvae.load_state_dict(best_sd_state)
    sdvae.eval()

    with torch.no_grad():
        xy_train_tensor = torch.tensor(xy_train_scaled, dtype=torch.float32).to(device)
        mu_prior, logvar_prior = sdvae.encode(xy_train_tensor)

    mudvae = MUDVAE(
        input_dim=input_dim_x,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        alpha=alpha_mu,
    ).to(device)

    train_dataset_mudvae = TensorDataset(
        torch.tensor(x_train_scaled, dtype=torch.float32),
        mu_prior.detach().cpu(),
        logvar_prior.detach().cpu(),
    )
    train_loader_mudvae = DataLoader(train_dataset_mudvae, batch_size=batch_size, shuffle=False)

    optimizer_mu = optim.AdamW(mudvae.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler_mu = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mu, mode="min", factor=0.5, patience=3
    )

    best_loss_mu = float("inf")
    best_mu_state = mudvae.state_dict()
    counter_mu = 0

    for epoch in range(max_epochs_mu):
        mudvae.train()
        epoch_loss = 0.0
        for x_batch, mu_batch, logvar_batch in train_loader_mudvae:
            x_batch = x_batch.to(device)
            mu_batch = mu_batch.to(device)
            logvar_batch = logvar_batch.to(device)

            optimizer_mu.zero_grad()
            recon_x, mu, logvar = mudvae(x_batch)
            loss = mudvae.loss_function(
                recon_x,
                x_batch,
                mu,
                logvar,
                mu_prior=mu_batch,
                logvar_prior=logvar_batch,
                kld_weight=0.01,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mudvae.parameters(), max_norm=1.0)
            optimizer_mu.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_dataset_mudvae)
        scheduler_mu.step(epoch_loss)

        if epoch_loss < best_loss_mu:
            best_loss_mu = epoch_loss
            best_mu_state = mudvae.state_dict()
            counter_mu = 0
        else:
            counter_mu += 1
            if counter_mu >= patience_mu:
                break

    mudvae.load_state_dict(best_mu_state)
    mudvae.eval()

    #оценка на тесте
    soft_sensor = SoftSensor(mudvae, sdvae).to(device)
    soft_sensor.eval()

    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
        y_pred_scaled = soft_sensor(x_test_tensor).cpu().numpy().reshape(-1, 1)

    xy_test_pred_scaled = np.hstack([x_test_scaled, y_pred_scaled])
    y_pred = xy_scaler.inverse_transform(xy_test_pred_scaled)[:, -1]

    y_true = xy_scaler.inverse_transform(xy_test_scaled)[:, -1]

    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    #Сохранение
    model_info = {
        "sdvae_state_dict": sdvae.state_dict(),
        "mudvae_state_dict": mudvae.state_dict(),
        "xy_scaler": xy_scaler,
        "x_scaler": x_scaler,
        "feature_names": feature_names,
        "target_col": SOFT_TARGET_COL,
        "hidden_dims": hidden_dims,
        "latent_dim": latent_dim,
        "input_dim_xy": input_dim_xy,
        "input_dim_x": input_dim_x,
    }

    SOFT_SD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model_info, SOFT_SD_MODEL_PATH)

    global _soft_sdvae_model
    _soft_sdvae_model = model_info

    logger.info(
        f"softsensor_sdvae_train: finished, rows_used={rows_used}, "
        f"n_features={len(feature_names)}, mse={mse:.4f}, r2={r2:.4f}, "
        f"model_path={SOFT_SD_MODEL_PATH}"
    )

    return SoftSensorSdvaeTrainResponse(
        rows_used=int(n_samples),
        n_features=len(feature_names),
        model_path=str(SOFT_SD_MODEL_PATH),
        mse=mse,
        r2=r2,
    )

@app.post("/softsensor_sdvae/predict_last", response_model=SoftSensorSdvaePredictResponse)
def softsensor_sdvae_predict_last(req: SoftSensorSdvaePredictRequest):
    """
    Предсказания SDVAE soft sensor на последних n_last строках
    """
    logger.info(f"softsensor_sdvae_predict_last: n_last={req.n_last}")
    model_info = load_softsensor_sdvae_model_from_disk()
    if model_info is None:
        raise HTTPException(
            status_code=400,
            detail="SDVAE soft sensor model is not trained yet",
        )

    feature_names: List[str] = model_info["feature_names"]
    target_col: str = model_info.get("target_col", SOFT_TARGET_COL)

    xy_scaler: MinMaxScaler = model_info["xy_scaler"]
    x_scaler: MinMaxScaler = model_info["x_scaler"]

    hidden_dims = model_info["hidden_dims"]
    latent_dim = model_info["latent_dim"]
    input_dim_xy = model_info["input_dim_xy"]
    input_dim_x = model_info["input_dim_x"]

    n_last = req.n_last if req.n_last > 0 else 1

    X_last, y_last = prepare_sdvae_last_data(n_last, feature_names)
    n_last = len(X_last)
    y_last = y_last.astype("float32")

    X_last_values = X_last.to_numpy(dtype="float32")

    X_last_scaled = x_scaler.transform(X_last_values)
    xy_last_scaled = xy_scaler.transform(
        np.hstack([X_last_values, y_last])
    )

    y_true = xy_scaler.inverse_transform(xy_last_scaled)[:, -1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sdvae = SDVAE(
        input_dim=input_dim_xy,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=0.1,
        alpha=2e-3,
    ).to(device)
    mudvae = MUDVAE(
        input_dim=input_dim_x,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=0.1,
        alpha=3e-4,
    ).to(device)

    sdvae.load_state_dict(model_info["sdvae_state_dict"])
    mudvae.load_state_dict(model_info["mudvae_state_dict"])
    sdvae.eval()
    mudvae.eval()

    soft_sensor = SoftSensor(mudvae, sdvae).to(device)
    soft_sensor.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(X_last_scaled, dtype=torch.float32).to(device)
        y_pred_scaled = soft_sensor(x_tensor).cpu().numpy().reshape(-1, 1)

    xy_last_pred_scaled = np.hstack([X_last_scaled, y_pred_scaled])
    y_pred = xy_scaler.inverse_transform(xy_last_pred_scaled)[:, -1]

    logger.info(
        f"softsensor_sdvae_predict_last: returned y_true={len(y_true)}, y_pred={len(y_pred)}"
    )

    return SoftSensorSdvaePredictResponse(
        n_last=int(n_last),
        y_true=[float(v) for v in y_true],
        y_pred=[float(v) for v in y_pred],
    )
