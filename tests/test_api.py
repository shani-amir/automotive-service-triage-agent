from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


ENDPOINT = "/api/v1/triage"
REVIEW = {"approved": True, "reviewer_id": "agent-7", "edited_reply_he": "נציג שירות יבדוק את הפנייה."}


def test_health_and_auto_complete(step3_graph_factory):
    graph, _ = step3_graph_factory()
    with TestClient(create_app(graph=graph)) as client:
        assert client.get("/health").json() == {
            "status": "ok", "service": "automotive-service-triage-agent", "hitl_enabled": True,
        }
        response = client.post(ENDPOINT, json={"message": "מהי מדיניות שירותי הדרך?"})
        assert response.status_code == 200
        body = response.json()
        assert UUID(body["thread_id"]).version == 4
        assert body["status"] == "auto_completed"
        assert body["decision"]["route"] == "auto_respond"
        assert body["erp_status"] == "skipped_no_plate"
        assert body["final_reply_he"] and body["internal_note_he"]
        assert body["cited_policy_titles"] == ["שירותי דרך"]
        assert client.get(f'{ENDPOINT}/{body["thread_id"]}').json()["is_paused"] is False
        assert client.post(f'{ENDPOINT}/{body["thread_id"]}/review', json=REVIEW).status_code == 409


def test_human_review_lifecycle(step3_graph_factory):
    graph, _ = step3_graph_factory(risk=True)
    with TestClient(create_app(graph=graph)) as client:
        response = client.post(ENDPOINT, json={"message": "תאונה. נדרש גרר.", "thread_id": "incident"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "awaiting_human_review"
        assert body["erp_status"] == "found"
        assert body["decision"]["requires_human_approval"] is True
        assert "angry_sentiment" in body["review_payload"]["reasons"]
        thread = f"{ENDPOINT}/incident"
        status = client.get(thread).json()
        assert status["is_paused"] is True
        assert status["status"] == "awaiting_human_review"
        assert status["final_reply_he"] is None
        assert client.post(ENDPOINT, json={"message": "Duplicate", "thread_id": "incident"}).status_code == 409
        result = client.post(f"{thread}/review", json=REVIEW)
        assert result.status_code == 200
        assert result.json() == {
            "thread_id": "incident", "status": "approved_by_human",
            "final_reply_he": REVIEW["edited_reply_he"], "human_review": {**REVIEW, "reviewer_note": None},
        }
        status = client.get(thread).json()
        assert status["is_paused"] is False
        assert status["status"] == "approved_by_human"
        assert client.post(f"{thread}/review", json=REVIEW).status_code == 409


def test_missing_customer_info(step3_graph_factory):
    graph, _ = step3_graph_factory(missing=True, drafting_error=True)
    with TestClient(create_app(graph=graph)) as client:
        response = client.post(ENDPOINT, json={"message": "נדרש גרר."})
        assert response.status_code == 200
        assert response.json()["status"] == "needs_customer_info"
        assert "מספר הרכב" in response.json()["final_reply_he"]


@pytest.mark.parametrize("message", ["", "  ", "customer@example.test" + "x" * 20_001])
def test_invalid_message_has_no_raw_pii(step3_graph_factory, message):
    graph, drafter = step3_graph_factory()
    with TestClient(create_app(graph=graph)) as client:
        result = client.post(ENDPOINT, json={"message": message})
        assert result.status_code in {400, 422}
        assert "customer@example.test" not in result.text
        drafter.assert_not_called()


def test_unknown_thread(step3_graph_factory):
    graph, _ = step3_graph_factory()
    with TestClient(create_app(graph=graph)) as client:
        assert client.get(f"{ENDPOINT}/unknown").status_code == 404
        assert client.post(f"{ENDPOINT}/unknown/review", json=REVIEW).status_code == 404
