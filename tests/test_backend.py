from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "CAUSAL-X"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analytics():
    response = client.get(
        "/analytics/overview?product=A"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["product"] == "A"
    assert data["average_price"] > 0
    assert data["average_demand"] > 0


def test_counterfactual():
    response = client.post(
        "/scenarios/counterfactual",
        json={
            "product": "A",
            "price_change_pct": -0.05,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "demand_change_pct" in data
    assert "profit_change_pct" in data


def test_decision_pipeline():
    response = client.post(
        "/decisions/analyze",
        json={
            "product": "A",
            "region": "North",
            "price_change": -0.05,
            "marketing_budget": 300000,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "causal_analysis" in data
    assert "optimization" in data
    assert "risk" in data
    assert "audit" in data
