from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.analyst_agent import AnalystAgent
from agents.orchestrator import CausalXOrchestrator
from causal.counterfactual import CounterfactualEngine


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(
    BASE_DIR / "data" / "raw" / "manufacturing_regional.csv"
)


app = FastAPI(
    title="CAUSAL-X API",
    description=(
        "AI-powered causal decision intelligence API "
        "for causal inference, counterfactual simulation, "
        "optimization, risk analysis, and decision auditing."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class DecisionRequest(BaseModel):
    product: str = Field(default="A")
    region: str = Field(default="North")
    price_change: float = Field(default=-0.05)
    marketing_budget: float = Field(default=300_000, ge=100_000)


class CounterfactualRequest(BaseModel):
    product: str = Field(default="A")
    price_change_pct: float = Field(default=-0.05)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "CAUSAL-X",
        "description": "AI-Powered Causal Decision Intelligence",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "causal-x-api",
    }


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

@app.get("/analytics/overview")
def analytics_overview(product: str = "A"):
    try:
        analyst = AnalystAgent(DATA_PATH)

        analysis = analyst.analyze_product(product)

        return {
            "product": product,
            "average_price": analysis.get("average_price"),
            "average_demand": analysis.get("average_demand"),
            "average_revenue": analysis.get("average_revenue"),
            "average_profit": analysis.get("average_profit"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analytics failed: {exc}",
        )


# ============================================================
# COMPLETE DECISION PIPELINE
# ============================================================

@app.post("/decisions/analyze")
def analyze_decision(request: DecisionRequest):
    try:
        orchestrator = CausalXOrchestrator(
            DATA_PATH
        )

        result = orchestrator.solve(
            product=request.product,
            region=request.region,
            price_change=request.price_change,
            marketing_budget=request.marketing_budget,
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Decision analysis failed: {exc}",
        )


# ============================================================
# COUNTERFACTUAL SCENARIO
# ============================================================

@app.post("/scenarios/counterfactual")
def counterfactual(request: CounterfactualRequest):
    try:
        engine = CounterfactualEngine(
            DATA_PATH
        )

        result = engine.summarize(
            request.product,
            request.price_change_pct,
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Counterfactual simulation failed: {exc}",
        )
