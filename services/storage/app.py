from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import yaml
import csv
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

LOG_PATH = Path(settings.get("log_path", "/app/logs/storage.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("storage")

STORAGE_FILE = Path(settings.get("storage_path", "/data/storage.csv"))

app = FastAPI(title="Storage Service")


class DataPoint(BaseModel):
    x: float
    y: float


class StoreRequest(BaseModel):
    items: List[DataPoint]


DATABASE: List[DataPoint] = []


def append_to_file(items: List[DataPoint]) -> None:
    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = STORAGE_FILE.exists()

    with STORAGE_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y"])
        if not file_exists:
            writer.writeheader()
        for item in items:
            writer.writerow({"x": item.x, "y": item.y})


@app.get("/health")
def health():
    items_count = len(DATABASE)
    file_exists = STORAGE_FILE.exists()
    logger.info(
        f"/health: items_in_memory={items_count}, file_exists={file_exists}"
    )
    return {
        "status": "ok",
        "service": "storage",
        "items": items_count,
        "file_exists": file_exists,
    }



@app.post("/store")
def store(req: StoreRequest):
    items = req.items
    logger.info(f"/store: received {len(items)} items")
    DATABASE.extend(items)
    append_to_file(items)
    logger.info(
        f"/store: total_in_memory={len(DATABASE)}, file={STORAGE_FILE}"
    )
    return {"stored": len(items), "total": len(DATABASE), "file": str(STORAGE_FILE)}



@app.get("/items", response_model=List[DataPoint])
def list_items(limit: int = 100):
    if limit <= 0:
        limit = 1
    result = DATABASE[-limit:]
    logger.info(f"/items: limit={limit}, returned={len(result)}")
    return result



@app.get("/stats")
def stats():
    file_exists = STORAGE_FILE.exists()
    file_size = STORAGE_FILE.stat().st_size if file_exists else 0
    in_memory = len(DATABASE)
    logger.info(
        f"/stats: in_memory={in_memory}, file_exists={file_exists}, "
        f"file_size_bytes={file_size}"
    )
    return {
        "in_memory": in_memory,
        "file_exists": file_exists,
        "file_path": str(STORAGE_FILE),
        "file_size_bytes": file_size,
    }

