import json
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent
RECSYS_DIR = BASE_DIR / "recsys"
DATA_DIR = RECSYS_DIR / "data"
RECS_DIR = RECSYS_DIR / "recommendations"

EVENTS_PATH = DATA_DIR / "events.parquet"
OFFLINE_RECS_PATH = RECS_DIR / "personal_als.parquet"


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Expected one of columns {candidates}, got {list(df.columns)}")


def get_test_users() -> tuple[int, int, int]:
    events = pd.read_parquet(EVENTS_PATH)
    personal = pd.read_parquet(OFFLINE_RECS_PATH)

    events_user_col = detect_column(events, ["user_id", "userid"])
    personal_user_col = detect_column(personal, ["user_id", "userid"])

    events_users = set(map(int, events[events_user_col].dropna().unique()))
    personal_users = set(map(int, personal[personal_user_col].dropna().unique()))

    user_with_personal_and_history = sorted(personal_users & events_users)[0]
    user_with_personal = sorted(personal_users)[1]

    user_without_personal = max(personal_users | events_users) + 1
    while user_without_personal in personal_users:
        user_without_personal += 1

    return user_without_personal, user_with_personal, user_with_personal_and_history


def print_response(title: str, response: requests.Response) -> None:
    print("=" * 70)
    print(title)
    print("status:", response.status_code)

    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(response.text)


def test_health() -> None:
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    print_response("HEALTHCHECK", r)


def test_user_without_personal_recs(user_id: int) -> None:
    payload = {"user_id": user_id, "k": 10}
    r = requests.post(f"{BASE_URL}/recommendations", json=payload, timeout=30)
    print_response("USER WITHOUT PERSONAL RECOMMENDATIONS", r)


def test_user_with_personal_recs(user_id: int) -> None:
    payload = {"user_id": user_id, "k": 10}
    r = requests.post(f"{BASE_URL}/recommendations", json=payload, timeout=30)
    print_response("USER WITH PERSONAL RECOMMENDATIONS", r)


def test_user_with_personal_and_history(user_id: int) -> None:
    payload = {"user_id": user_id, "k": 10}
    r = requests.post(f"{BASE_URL}/recommendations", json=payload, timeout=30)
    print_response("USER WITH PERSONAL RECOMMENDATIONS AND ONLINE HISTORY", r)


if __name__ == "__main__":
    user_without_personal, user_with_personal, user_with_personal_and_history = get_test_users()

    print(f"Selected user without personal recommendations: {user_without_personal}")
    print(f"Selected user with personal recommendations: {user_with_personal}")
    print(f"Selected user with personal recommendations and online history: {user_with_personal_and_history}")

    test_health()
    test_user_without_personal_recs(user_without_personal)
    test_user_with_personal_recs(user_with_personal)
    test_user_with_personal_and_history(user_with_personal_and_history)
    