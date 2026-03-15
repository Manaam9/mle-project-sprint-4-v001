import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def print_response(title, response):
    print("=" * 70)
    print(title)
    print("status:", response.status_code)

    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print_response("HEALTHCHECK", r)


def test_user_without_personal_recs():
    payload = {
        "user_id": 999999,
        "k": 10
    }

    r = requests.post(f"{BASE_URL}/recommendations", json=payload)
    print_response("USER WITHOUT PERSONAL RECOMMENDATIONS", r)


def test_user_with_personal_recs():
    payload = {
        "user_id": 3,
        "k": 10
    }

    r = requests.post(f"{BASE_URL}/recommendations", json=payload)
    print_response("USER WITH PERSONAL RECOMMENDATIONS", r)


def test_user_with_personal_and_history():
    payload = {
        "user_id": 4,
        "k": 10
    }

    r = requests.post(f"{BASE_URL}/recommendations", json=payload)
    print_response("USER WITH PERSONAL RECOMMENDATIONS AND ONLINE HISTORY", r)


if __name__ == "__main__":
    test_health()
    test_user_without_personal_recs()
    test_user_with_personal_recs()
    test_user_with_personal_and_history()
    