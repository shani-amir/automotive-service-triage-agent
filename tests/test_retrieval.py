import sqlite3
from unittest.mock import Mock

import pytest
from langsmith import tracing_context

from src.agent.graph import build_triage_graph
from src.schemas.triage import TriageExtraction
from src.services.erp_repo import SQLiteVehicleRepository, normalize_license_plate
from src.services.policy_rag import LocalHybridPolicyRetriever


def test_sqlite_plate_lookup_and_idempotent_seed(tmp_path):
    db_path = tmp_path / "erp" / "vehicles.db"
    repository = SQLiteVehicleRepository(db_path)
    assert normalize_license_plate(" 482 - 91\u2013302 ") == "48291302"
    vehicle = repository.get_by_license_plate("482-91-302")
    assert vehicle is not None
    assert vehicle == repository.get_by_license_plate("48291302")
    assert vehicle == SQLiteVehicleRepository(db_path).get_by_license_plate("48291302")
    assert vehicle.contract_type == "operational_leasing"
    assert vehicle.contract_status == "active"
    assert vehicle.includes_towing and vehicle.includes_replacement_car
    assert vehicle.deductible_nis == 1500
    assert repository.get_by_license_plate("999-88-777") is None
    assert repository.get_by_license_plate("unknown") is None


@pytest.mark.parametrize("error", [sqlite3.OperationalError("database is locked"), TimeoutError()])
def test_erp_retries_transient_errors(monkeypatch, error):
    repository = SQLiteVehicleRepository(":memory:", retry_backoff=0)
    record = repository.get_by_license_plate("48291302")
    assert record is not None
    lookup = Mock(side_effect=[error, record])
    monkeypatch.setattr(repository, "_lookup", lookup)
    assert repository.get_by_license_plate("482-91-302") == record
    assert lookup.call_count == 2


def _invoke(plate, repository, retriever, *, raw_input="צריך גרר וגם רכב חלופי", summary="נדרש גרר."):
    llm = Mock()
    llm.invoke.return_value = TriageExtraction(
        license_plate=plate, intent="accident_towing", urgency="high",
        sentiment="neutral", missing_info=[], summary_he=summary,
    )
    graph = build_triage_graph(llm=llm, vehicle_repo=repository, policy_retriever=retriever)
    with tracing_context(enabled=False):
        return graph.invoke({"raw_input": raw_input})


def test_exhausted_erp_retries_continue_to_policy_retrieval(monkeypatch):
    repository = SQLiteVehicleRepository(":memory:", max_retries=2, retry_backoff=0)
    lookup = Mock(side_effect=sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(repository, "_lookup", lookup)
    result = _invoke("482-91-302", repository, LocalHybridPolicyRetriever())
    assert lookup.call_count == 3
    assert result["erp_status"] == "error"
    assert result["vehicle_context"] is None
    assert result["retrieval_filters"]["contract_type"] is None
    assert result["retrieved_policies"]


def test_no_plate_skips_erp_and_retrieves_general_policy():
    repository = Mock()
    result = _invoke(None, repository, LocalHybridPolicyRetriever())
    repository.get_by_license_plate.assert_not_called()
    assert result["erp_status"] == "skipped_no_plate"
    assert result["vehicle_context"] is None
    assert result["retrieval_filters"]["contract_type"] is None
    assert any("all" in chunk.applies_to for chunk in result["retrieved_policies"])


@pytest.mark.parametrize(("plate", "status", "contract_type"), [
    ("482-91-302", "found", "operational_leasing"),
    ("999-88-777", "not_found", None),
])
def test_graph_passes_contract_filter_to_retriever(plate, status, contract_type):
    retriever = Mock(wraps=LocalHybridPolicyRetriever())
    result = _invoke(plate, SQLiteVehicleRepository(":memory:"), retriever)
    assert result["erp_status"] == status
    assert result["retrieval_filters"]["contract_type"] == contract_type
    assert retriever.retrieve.call_args.kwargs == {
        "contract_type": contract_type, "intent": "accident_towing",
    }
    assert result["retrieved_policies"]


def test_policy_metadata_excludes_other_contracts_and_ranks_relevant_chunks():
    retriever = LocalHybridPolicyRetriever()
    query = "צריך גרר ורכב חלופי בליסינג תפעולי, האם יש אחריות יצרן?"
    unfiltered = retriever.retrieve(query, top_k=20)
    assert any(chunk.applies_to == ["manufacturer_warranty"] for chunk in unfiltered)
    filtered = retriever.retrieve(query, "operational_leasing", "accident_towing", top_k=20)
    assert all({"all", "operational_leasing"}.intersection(chunk.applies_to) for chunk in filtered)
    assert {chunk.doc_id for chunk in filtered[:2]} == {"towing_and_roadside", "replacement_car"}
    assert all("operational_leasing" in chunk.applies_to for chunk in filtered[:2])


@pytest.mark.parametrize(("query", "doc_id"), [
    ("לגרר", "towing_and_roadside"),
    ("תאונת", "accidents_and_deductible"),
    ("מבוטחת", "accidents_and_deductible"),
])
def test_policy_matches_hebrew_inflections(query, doc_id):
    chunks = LocalHybridPolicyRetriever().retrieve(query)
    assert chunks[0].doc_id == doc_id


def test_retrieval_masks_pii_in_source_and_generated_summary():
    retriever = Mock(wraps=LocalHybridPolicyRetriever())
    pii = ("123456782", "052-555-0199", "driver@example.test", "4111 1111 1111 1111")
    result = _invoke(
        None, Mock(), retriever,
        raw_input=f"צריך גרר. {' '.join(pii)}",
        summary=f"נדרש גרר. {' '.join(pii)}",
    )
    (query,), _ = retriever.retrieve.call_args
    for value in pii:
        assert value not in query
        assert value not in result["retrieval_filters"]["query_preview"]
    assert query.count("[ISRAELI_ID]") == 2
    assert query.count("[PHONE]") == 2
    assert "נדרש גרר" in query and "צריך גרר" in query
    assert "accident_towing" in query
