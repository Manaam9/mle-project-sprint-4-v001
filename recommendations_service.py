from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent

RECSYS_DIR = BASE_DIR / "recsys"
DATA_DIR = RECSYS_DIR / "data"
RECS_DIR = RECSYS_DIR / "recommendations"

EVENTS_PATH = DATA_DIR / "events.parquet"
OFFLINE_RECS_PATH = RECS_DIR / "personal_als.parquet"
POPULAR_RECS_PATH = RECS_DIR / "top_popular.parquet"

DEFAULT_K = 10
ONLINE_WEIGHT = 1.0
OFFLINE_WEIGHT = 0.7

app = FastAPI(title="Recommendations Service")


class RecommendationRequest(BaseModel):
    user_id: int
    k: int = DEFAULT_K


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    strategy: str


def detect_column(df: pd.DataFrame, candidates: List[str], file_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Не найдена нужная колонка в {file_name}. "
        f"Ожидалась одна из: {candidates}. "
        f"Доступные колонки: {list(df.columns)}"
    )


def load_offline_recs() -> Dict[int, List[int]]:
    if not OFFLINE_RECS_PATH.exists():
        return {}

    df = pd.read_parquet(OFFLINE_RECS_PATH)

    user_col = detect_column(
        df, ["user_id", "userid"], OFFLINE_RECS_PATH.name
    )
    item_col = detect_column(
        df, ["item_id", "track_id", "itemid"], OFFLINE_RECS_PATH.name
    )

    if "rank" in df.columns:
        df = df.sort_values([user_col, "rank"])
    elif "score" in df.columns:
        df = df.sort_values([user_col, "score"], ascending=[True, False])

    return (
        df.groupby(user_col)[item_col]
        .apply(list)
        .to_dict()
    )


def load_popular_recs() -> List[int]:
    if not POPULAR_RECS_PATH.exists():
        return []

    df = pd.read_parquet(POPULAR_RECS_PATH)

    item_col = detect_column(
        df, ["item_id", "track_id", "itemid"], POPULAR_RECS_PATH.name
    )

    if "rank" in df.columns:
        df = df.sort_values("rank")
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    return df[item_col].drop_duplicates().tolist()


def load_online_history() -> Dict[int, List[int]]:
    if not EVENTS_PATH.exists():
        return {}

    df = pd.read_parquet(EVENTS_PATH)

    user_col = detect_column(
        df, ["user_id", "userid"], EVENTS_PATH.name
    )
    item_col = detect_column(
        df, ["item_id", "track_id", "itemid"], EVENTS_PATH.name
    )

    if "timestamp" in df.columns:
        df = df.sort_values([user_col, "timestamp"])
    elif "event_date" in df.columns:
        df = df.sort_values([user_col, "event_date"])

    return (
        df.groupby(user_col)[item_col]
        .apply(list)
        .to_dict()
    )


OFFLINE_RECS = load_offline_recs()
POPULAR_RECS = load_popular_recs()
ONLINE_HISTORY = load_online_history()


def build_online_candidates(history_items: List[int], popular_items: List[int]) -> List[int]:
    seen_items = set(history_items)
    return [item for item in popular_items if item not in seen_items]


def merge_recommendations(
    offline_items: List[int],
    online_items: List[int],
    popular_items: List[int],
    history_items: List[int],
    k: int,
) -> List[int]:
    scores: Dict[int, float] = {}
    seen_items = set(history_items)

    for rank, item_id in enumerate(online_items, start=1):
        if item_id not in seen_items:
            scores[item_id] = scores.get(item_id, 0.0) + ONLINE_WEIGHT / rank

    for rank, item_id in enumerate(offline_items, start=1):
        if item_id not in seen_items:
            scores[item_id] = scores.get(item_id, 0.0) + OFFLINE_WEIGHT / rank

    ranked_items = [
        item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    result: List[int] = []
    used: Set[int] = set()

    for item_id in ranked_items:
        if item_id not in used:
            result.append(item_id)
            used.add(item_id)
        if len(result) >= k:
            return result

    for item_id in popular_items:
        if item_id not in used and item_id not in seen_items:
            result.append(item_id)
            used.add(item_id)
        if len(result) >= k:
            break

    return result


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    user_id = request.user_id
    k = request.k

    if k <= 0:
        raise HTTPException(status_code=400, detail="Параметр k должен быть положительным")

    offline_items = OFFLINE_RECS.get(user_id, [])
    history_items = ONLINE_HISTORY.get(user_id, [])
    online_items = build_online_candidates(history_items, POPULAR_RECS)

    if not offline_items:
        strategy = "popular_fallback"
        recommendations = [
            item for item in POPULAR_RECS if item not in set(history_items)
        ][:k]
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            strategy=strategy,
        )

    if not history_items:
        strategy = "offline_only"
        recommendations = [
            item for item in offline_items if item not in set(history_items)
        ][:k]
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            strategy=strategy,
        )

    strategy = "hybrid_online_offline"
    recommendations = merge_recommendations(
        offline_items=offline_items,
        online_items=online_items,
        popular_items=POPULAR_RECS,
        history_items=history_items,
        k=k,
    )

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        strategy=strategy,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "recommendations_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
    