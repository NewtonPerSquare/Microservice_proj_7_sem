from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from pathlib import Path
from datetime import datetime
import yaml
import requests


CONFIG_PATH = Path(__file__).with_name("config.yaml")


DEFAULT_SETTINGS = {
    "collector_url": "http://collector:8000",
    "storage_url": "http://storage:8000",
    "ml_url": "http://ml_service:8000",
    "request_timeout": 60.0,
}


def load_config() -> dict:
    cfg = DEFAULT_SETTINGS.copy()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                cfg.update(data)
    return cfg


settings = load_config()

app = FastAPI(title="Web Master Service")


class DemoRequest(BaseModel):
    batch_size: int = 5


class DemoResponse(BaseModel):
    collected: int
    stored_total: int
    predictions: List[float]
    used_trained_model: bool


class TrainModelResponse(BaseModel):
    rows_used: int
    model_path: str


class SoftSensorTrainResponse(BaseModel):
    rows_used: int
    n_features: int
    model_path: str
    mse: float
    r2: float


class SoftSensorEvalRequest(BaseModel):
    n_last: int = 50


class SoftSensorEvalResponse(BaseModel):
    n_last: int
    y_true: List[float]
    y_pred: List[float]


class SoftSensorSdvaeTrainResponse(BaseModel):
    rows_used: int
    n_features: int
    model_path: str
    mse: float
    r2: float


class SoftSensorSdvaeEvalRequest(BaseModel):
    n_last: int = 50


class SoftSensorSdvaeEvalResponse(BaseModel):
    n_last: int
    y_true: List[float]
    y_pred: List[float]


@app.get("/health")
def health():
    return {"status": "ok", "service": "web_master"}


@app.post("/scenario/demo", response_model=DemoResponse)
def run_demo(req: DemoRequest):
    """
    Демонстрационный сценарий:
    1)запросить батч у Collector
    2)сохранить в Storage
    3)отправить в MLService и вернуть предсказания
    """

    #1. Collector
    try:
        r = requests.post(
            f"{settings['collector_url']}/batch",
            json={"size": req.batch_size},
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        batch = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Collector error: {e}")

    items = batch.get("items", [])
    if not isinstance(items, list):
        raise HTTPException(status_code=500, detail="Collector returned invalid data format")

    #2. Storage
    try:
        r = requests.post(
            f"{settings['storage_url']}/store",
            json={"items": items},
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        store_resp = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Storage error: {e}")

    stored_total = int(store_resp.get("total", len(items)))

    #3. MLService (простая модель)
    try:
        r = requests.post(
            f"{settings['ml_url']}/predict",
            json={"items": items},
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        pred_resp = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLService error: {e}")

    predictions = pred_resp.get("y_pred", [])
    used_trained_model = bool(pred_resp.get("used_trained_model", False))

    if not isinstance(predictions, list):
        raise HTTPException(status_code=500, detail="MLService returned invalid data format")

    return DemoResponse(
        collected=len(items),
        stored_total=stored_total,
        predictions=predictions,
        used_trained_model=used_trained_model,
    )


@app.post("/scenario/train_model", response_model=TrainModelResponse)
def train_model():
    try:
        r = requests.post(
            f"{settings['ml_url']}/train_from_file",
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        resp = r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=r.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLService train error: {e}")

    return TrainModelResponse(
        rows_used=int(resp.get("rows_used", 0)),
        model_path=str(resp.get("model_path", "")),
    )


@app.post("/scenario/softsensor_train", response_model=SoftSensorTrainResponse)
def softsensor_train():
    try:
        r = requests.post(
            f"{settings['ml_url']}/softsensor/train",
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        resp = r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=r.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLService softsensor train error: {e}")

    return SoftSensorTrainResponse(
        rows_used=int(resp.get("rows_used", 0)),
        n_features=int(resp.get("n_features", 0)),
        model_path=str(resp.get("model_path", "")),
        mse=float(resp.get("mse", 0.0)),
        r2=float(resp.get("r2", 0.0)),
    )


@app.post("/scenario/softsensor_eval", response_model=SoftSensorEvalResponse)
def softsensor_eval(req: SoftSensorEvalRequest):
    try:
        r = requests.post(
            f"{settings['ml_url']}/softsensor/predict_last",
            json={"n_last": req.n_last},
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        resp = r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=r.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLService softsensor eval error: {e}")

    return SoftSensorEvalResponse(
        n_last=int(resp.get("n_last", 0)),
        y_true=[float(v) for v in resp.get("y_true", [])],
        y_pred=[float(v) for v in resp.get("y_pred", [])],
    )


@app.post("/scenario/softsensor_sdvae_train", response_model=SoftSensorSdvaeTrainResponse)
def softsensor_sdvae_train():

    try:
        r = requests.post(
            f"{settings['ml_url']}/softsensor_sdvae/train",
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        resp = r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=r.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLService softsensor_sdvae train error: {e}")

    return SoftSensorSdvaeTrainResponse(
        rows_used=int(resp.get("rows_used", 0)),
        n_features=int(resp.get("n_features", 0)),
        model_path=str(resp.get("model_path", "")),
        mse=float(resp.get("mse", 0.0)),
        r2=float(resp.get("r2", 0.0)),
    )


@app.post("/scenario/softsensor_sdvae_eval", response_model=SoftSensorSdvaeEvalResponse)
def softsensor_sdvae_eval(req: SoftSensorSdvaeEvalRequest):
    try:
        r = requests.post(
            f"{settings['ml_url']}/softsensor_sdvae/predict_last",
            json={"n_last": req.n_last},
            timeout=settings["request_timeout"],
        )
        r.raise_for_status()
        resp = r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=r.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLService softsensor_sdvae eval error: {e}")

    return SoftSensorSdvaeEvalResponse(
        n_last=int(resp.get("n_last", 0)),
        y_true=[float(v) for v in resp.get("y_true", [])],
        y_pred=[float(v) for v in resp.get("y_pred", [])],
    )



@app.get("/report/softsensor", response_class=HTMLResponse)
def softsensor_report(n_last: int = 100):
    if n_last <= 0:
        n_last = 100

    ml_url = settings["ml_url"].rstrip("/")
    timeout = settings.get("request_timeout", 30.0)

    try:
        r = requests.post(
            f"{ml_url}/softsensor/predict_last",
            json={"n_last": n_last},
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"MLService softsensor_predict_last error: {e}",
        )

    data = r.json()
    y_true = data.get("y_true", [])
    y_pred = data.get("y_pred", [])

    if not y_true or not y_pred:
        raise HTTPException(
            status_code=500,
            detail="Empty y_true/y_pred in MLService response",
        )

    n = min(len(y_true), len(y_pred))
    y_true = [float(v) for v in y_true[:n]]
    y_pred = [float(v) for v in y_pred[:n]]

    errors = [yt - yp for yt, yp in zip(y_true, y_pred)]
    mse = sum(e * e for e in errors) / n

    mean_y = sum(y_true) / n
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчёт по soft sensor (RandomForest)</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      margin: 32px;
      background-color: #f5f5f5;
    }}
    .card {{
      background: #ffffff;
      border-radius: 12px;
      padding: 20px;
      max-width: 800px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    h1 {{ margin-top: 0; }}
    dt {{ font-weight: 600; }}
    dd {{ margin: 0 0 8px 0; }}
    .small {{ color: #6b7280; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Отчёт по soft sensor (RandomForest)</h1>
    <p class="small">Сгенерировано: {generated_at}</p>
    <dl>
      <dt>Число точек для оценки</dt>
      <dd>{n}</dd>
      <dt>MSE</dt>
      <dd>{mse:.4f}</dd>
      <dt>R²</dt>
      <dd>{r2:.4f}</dd>
    </dl>
    <p class="small">
      Источник: MLService /softsensor/predict_last, n_last={n_last}.
    </p>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
