from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import yaml
import random
import pandas as pd
import logging


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    return {}


settings = load_config()
LOG_PATH = Path(settings.get("log_path", "/app/logs/collector.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("collector")


app = FastAPI(title="Collector Service")

DATASET: Optional[pd.DataFrame] = None
CURRENT_INDEX: int = 0


class BatchRequest(BaseModel):
    size: int = 10  #размер батча


class DataPoint(BaseModel):
    x: float
    y: float


class BatchResponse(BaseModel):
    items: List[DataPoint]


def load_dataset() -> Optional[pd.DataFrame]:
    path_str = settings.get("data_path", "/data/collector_source.csv")
    path = Path(path_str)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns):
        return None

    df = df[["x", "y"]].dropna().reset_index(drop=True)
    if df.empty:
        return None
    return df


@app.on_event("startup")
def on_startup():
    global DATASET, CURRENT_INDEX
    DATASET = load_dataset()
    CURRENT_INDEX = 0


@app.get("/health")
def health():
    has_dataset = DATASET is not None
    dataset_len = int(len(DATASET)) if has_dataset else 0
    logger.info(f"/health: has_dataset={has_dataset}, dataset_len={dataset_len}")
    return {
        "status": "ok",
        "service": "collector",
        "has_dataset": has_dataset,
        "dataset_len": dataset_len,
    }



@app.post("/batch", response_model=BatchResponse)
def get_batch(req: BatchRequest):
    """
    Если есть CSV-датасет — выдаём батчи по порядку.
    Когда доходим до конца — начинаем с начала.
    Если датасета нет — генерируем синтетические данные.
    """
    size = req.size if req.size > 0 else 1
    logger.info(f"/batch: requested size={size}, has_dataset={DATASET is not None}")

    #есть загруженный датасет
    global CURRENT_INDEX
    if DATASET is not None:
        start = CURRENT_INDEX
        end = start + size

        if start >= len(DATASET):
            #если дошли до конца — начинаем сначала
            start = 0
            end = size

        if end > len(DATASET):
            end = len(DATASET)

        logger.info(
            f"/batch: using dataset rows {start}:{end} (len={len(DATASET)})"
        )

        batch_df = DATASET.iloc[start:end]
        CURRENT_INDEX = end

        items = [
            DataPoint(x=float(row["x"]), y=float(row["y"]))
            for _, row in batch_df.iterrows()
        ]
        logger.info(f"/batch: returned {len(items)} items (dataset)")
        return BatchResponse(items=items)

    #датасета нет — синтетика
    items: List[DataPoint] = []
    for _ in range(size):
        x = random.random()
        y = 2.0 * x + 0.1
        items.append(DataPoint(x=x, y=y))

    logger.info(f"/batch: returned {len(items)} items (synthetic)")
    return BatchResponse(items=items)

