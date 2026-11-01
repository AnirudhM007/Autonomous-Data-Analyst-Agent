import io

from fastapi.testclient import TestClient
from openpyxl import Workbook

from app.main import app, datasets, analyses


client = TestClient(app)


def setup_function():
    datasets.clear()
    analyses.clear()


def test_health_and_example_ingestion():
    assert client.get("/api/health").json()["status"] == "ok"
    response = client.post("/api/datasets/examples/sample-sales")
    assert response.status_code == 200
    dataset = response.json()
    assert dataset["rows"] == 720
    profile = client.get(f"/api/datasets/{dataset['id']}/profile").json()
    assert "revenue" in profile["column_names"]


def test_upload_validation_and_analysis_lifecycle():
    response = client.post("/api/datasets/upload", files={"file": ("tiny.csv", b"region,revenue\nNorth,10\nSouth,15\n", "text/csv")})
    assert response.status_code == 200
    dataset_id = response.json()["id"]
    started = client.post("/api/analysis/start", json={"dataset_id": dataset_id, "objective": "Compare revenue by region"})
    assert started.status_code == 202
    analysis_id = started.json()["analysis_id"]
    result = client.get(f"/api/analysis/{analysis_id}")
    assert result.status_code == 200
    assert result.json()["status"] == "complete"
    assert client.get(f"/api/analysis/{analysis_id}/trace").json()["events"]
    assert client.get(f"/api/analysis/{analysis_id}/report").json()["executive_summary"]
    assert client.get(f"/api/analysis/{analysis_id}/report?format=csv").headers["content-type"].startswith("text/csv")
    assert client.get(f"/api/analysis/{analysis_id}/report?format=pdf").headers["content-type"] == "application/pdf"


def test_upload_rejects_unsupported_extension():
    response = client.post("/api/datasets/upload", files={"file": ("script.py", b"print(1)", "text/plain")})
    assert response.status_code == 415


def test_selected_provider_requires_api_key():
    dataset_id = client.post("/api/datasets/examples/sample-sales").json()["id"]
    response = client.post("/api/analysis/start", json={
        "dataset_id": dataset_id, "provider": "gemini"
    })
    assert response.status_code == 422
    assert response.json()["detail"] == "Enter an API key for the selected provider."


def test_user_key_is_passed_to_run_but_never_saved(monkeypatch):
    import app.main as main

    dataset_id = client.post("/api/datasets/examples/sample-sales").json()["id"]
    captured = {}

    def capture(analysis_id, provider, api_key):
        captured.update(analysis_id=analysis_id, provider=provider, api_key=api_key)

    monkeypatch.setattr(main, "_execute_analysis", capture)
    response = client.post("/api/analysis/start", json={
        "dataset_id": dataset_id, "provider": "openai", "api_key": "test-user-secret"
    })
    assert response.status_code == 202
    record = analyses[response.json()["analysis_id"]]
    assert captured == {"analysis_id": record["id"], "provider": "openai", "api_key": "test-user-secret"}
    assert "api_key" not in record
    assert "test-user-secret" not in str(record)


def test_xlsx_upload_reads_sheet_values():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["region", "revenue"])
    sheet.append(["North", 21])
    output = io.BytesIO()
    workbook.save(output)
    response = client.post("/api/datasets/upload", files={
        "file": ("tiny.xlsx", output.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    })
    assert response.status_code == 200
    assert response.json()["rows"] == 1
    profile = client.get(f"/api/datasets/{response.json()['id']}/profile").json()
    assert profile["column_names"] == ["region", "revenue"]
