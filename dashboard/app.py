import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from agents.orchestrator import (
    CausalXOrchestrator
)

from agents.llm.copilot import (
    DecisionCopilot
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CAUSAL-X",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = (
    "data/raw/manufacturing_regional.csv"
)


@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


@st.cache_resource
def load_orchestrator():

    return CausalXOrchestrator(
        DATA_PATH
    )


df = load_data()

orchestrator = (
    load_orchestrator()
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "CAUSAL-X"
)

st.subheader(
    "AI-Powered Causal Decision Intelligence"
)

st.markdown(
    """
CAUSAL-X combines **causal inference, forecasting,
counterfactual simulation, risk analysis, and optimization**
to help decision-makers understand not only *what may happen*,
but *what would happen if we changed something*.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Decision Controls"
)

products = sorted(
    df["product"].unique()
)

regions = sorted(
    df["region"].unique()
)

product = st.sidebar.selectbox(
    "Product",
    products
)

region = st.sidebar.selectbox(
    "Region",
    regions
)

price_change = st.sidebar.slider(
    "Price intervention (%)",
    min_value=-10.0,
    max_value=10.0,
    value=-5.0,
    step=0.5
)

marketing_budget = st.sidebar.number_input(
    "Maximum marketing budget (₹)",
    min_value=0,
    max_value=1_000_000,
    value=300_000,
    step=50_000
)

run_analysis = st.sidebar.button(
    "RUN ANALYSIS",
    type="primary",
    width="stretch"
)


# ============================================================
# INITIAL STATE
# ============================================================

if "result" not in st.session_state:

    st.session_state.result = None


# ============================================================
# RUN PIPELINE
# ============================================================

if run_analysis:

    with st.spinner(
        "Running CAUSAL-X decision pipeline..."
    ):

        try:

            result = (
                orchestrator.solve(
                    product=product,
                    region=region,
                    price_change=
                        price_change / 100,
                    marketing_budget=
                        marketing_budget
                )
            )

            st.session_state.result = result

        except Exception as error:

            st.error(
                f"Pipeline error: {error}"
            )

            st.stop()


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result is None:

    st.info(
        "Select a product and region, "
        "then click RUN ANALYSIS."
    )

    st.stop()


causal = result[
    "causal_analysis"
]

optimization = result[
    "optimization"
]

risk = result[
    "risk"
]

audit = result[
    "audit"
]

business = result[
    "business_analysis"
]

forecast = result[
    "forecast"
]


# ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.header(
    "Business Overview"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Price",
        f"₹{business['average_price']:,.0f}"
    )

with col2:

    st.metric(
        "Average Demand",
        f"{business['average_demand']:,.0f}"
    )

with col3:

    st.metric(
        "Average Revenue",
        f"₹{business['average_revenue']:,.0f}"
    )

with col4:

    st.metric(
        "Average Profit",
        f"₹{business['average_profit']:,.0f}"
    )


st.divider()


# ============================================================
# CAUSAL EFFECT
# ============================================================

st.header(
    "1. Causal Effect"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Elasticity",
        f"{causal['estimated_elasticity']:.3f}"
    )

with c2:

    st.metric(
        "95% CI",
        (
            f"["
            f"{causal['bootstrap_ci_lower']:.2f}, "
            f"{causal['bootstrap_ci_upper']:.2f}"
            f"]"
        )
    )

with c3:

    st.metric(
        "Placebo Effect",
        f"{causal['placebo_elasticity']:.3f}"
    )

with c4:

    robustness = causal[
        "robustness"
    ]

    if robustness == "HIGH":

        st.success(
            f"Robustness: {robustness}"
        )

    elif robustness == "MEDIUM":

        st.warning(
            f"Robustness: {robustness}"
        )

    else:

        st.error(
            f"Robustness: {robustness}"
        )


st.caption(
    "Causal estimate is interpreted under the "
    "model's identification assumptions."
)


# ============================================================
# COUNTERFACTUAL
# ============================================================

st.header(
    "2. Counterfactual Simulation"
)

counterfactual = (
    causal[
        "counterfactual_demand"
    ]
)

baseline = (
    causal[
        "baseline_demand"
    ]
)

demand_change = (
    (
        counterfactual
        - baseline
    )
    / baseline
    * 100
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Baseline Demand",
        f"{baseline:,.0f}"
    )

with c2:

    st.metric(
        "Counterfactual Demand",
        f"{counterfactual:,.0f}"
    )

with c3:

    st.metric(
        "Demand Effect",
        f"{demand_change:+.2f}%"
    )


# ============================================================
# DEMAND VISUALIZATION
# ============================================================

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[
            "Observed",
            "Counterfactual"
        ],

        y=[
            baseline,
            counterfactual
        ],

        text=[
            f"{baseline:,.0f}",
            f"{counterfactual:,.0f}"
        ],

        textposition="auto",
    )
)

fig.update_layout(
    title=(
        "What-if: Demand under "
        f"{price_change:+.1f}% Price Change"
    ),

    yaxis_title="Demand",

    height=400,

    showlegend=False,
)

st.plotly_chart(
    fig,
    width="stretch"
)


st.divider()


# ============================================================
# USER SCENARIO
# ============================================================

st.header(
    "3. Your Scenario"
)

scenario_price = (
    business["average_price"]
    * (1 + price_change / 100)
)

s1, s2, s3 = st.columns(3)

with s1:

    st.metric(
        "Scenario Price",
        f"₹{scenario_price:,.0f}"
    )

with s2:

    st.metric(
        "Marketing Budget",
        f"₹{marketing_budget:,.0f}"
    )

with s3:

    st.metric(
        "Scenario Demand",
        f"{counterfactual:,.0f}"
    )

st.caption(
    "This is the outcome of the intervention you specified. "
    "The AI optimizer below searches for a potentially better "
    "risk-adjusted decision."
)

st.divider()


# ============================================================
# OPTIMIZATION
# ============================================================

st.header(
    "4. Risk-Adjusted Optimization"
)

o1, o2, o3, o4 = st.columns(4)

with o1:

    st.metric(
        "Recommended Price",
        f"₹{optimization['price']:,.0f}"
    )

with o2:

    st.metric(
        "Price Change",
        f"{optimization['price_change'] * 100:+.1f}%"
    )

with o3:

    st.metric(
        "Expected Demand",
        f"{optimization['expected_demand']:,.0f}"
    )

with o4:

    st.metric(
        "Expected Profit",
        f"₹{optimization['expected_profit']:,.0f}"
    )

o5, o6, o7 = st.columns(3)

with o5:

    st.metric(
        "5th Percentile Profit",
        f"₹{optimization['profit_p05']:,.0f}"
    )

with o6:

    st.metric(
        "Downside CVaR 95%",
        f"₹{optimization['cvar_95']:,.0f}"
    )

with o7:

    st.metric(
        "Risk-Adjusted Objective",
        f"₹{optimization['objective']:,.0f}"
    )



# ============================================================
# SCENARIO EXPLORER
# ============================================================

st.header(
    "5. Scenario Explorer"
)

st.markdown(
    """
Explore how price interventions affect demand and profit.
The optimizer evaluates the business decision while the
causal model isolates the estimated effect of changing price.
"""
)

decisions = optimization.get(
    "all_decisions"
)

if (
    decisions is not None
    and not decisions.empty
):

    scenario_df = decisions.copy()

    # ---------------------------------------------------------
    # PRICE INTERVENTION
    # ---------------------------------------------------------

    scenario_df[
        "price_change_pct"
    ] = (
        scenario_df[
            "price_change"
        ] * 100
    )

    # ---------------------------------------------------------
    # BEST MARKETING DECISION AT EACH PRICE
    # ---------------------------------------------------------

    scenario_df = (
        scenario_df
        .sort_values(
            "objective",
            ascending=False
        )
        .groupby(
            "price_change_pct",
            as_index=False
        )
        .first()
        .sort_values(
            "price_change_pct"
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # OPTIMAL DECISION
    # ---------------------------------------------------------

    optimum_change = (
        optimization[
            "price_change"
        ] * 100
    )

    optimum_price = (
        optimization[
            "price"
        ]
    )

    optimum_demand = (
        optimization[
            "expected_demand"
        ]
    )

    optimum_profit = (
        optimization[
            "expected_profit"
        ]
    )

    # =========================================================
    # CAUSAL MODEL
    # =========================================================

    causal_result = result.get(
        "causal_analysis",
        {}
    )

    causal_elasticity = causal_result.get(
        "estimated_elasticity",
        optimization.get(
            "elasticity",
            -1.0
        )
    )

    business_result = result.get(
        "business_analysis",
        {}
    )

    causal_baseline = causal_result.get(
        "baseline_demand",
        business_result.get(
            "average_demand",
            0
        )
    )

    # ---------------------------------------------------------
    # PURE CAUSAL RESPONSE
    # ---------------------------------------------------------

    causal_price_changes = np.arange(
        -0.10,
        0.101,
        0.01
    )

    causal_demands = (
        causal_baseline
        * (
            1 + causal_price_changes
        )
        ** causal_elasticity
    )

    # =========================================================
    # 1. DEMAND RESPONSE
    # =========================================================

    st.subheader(
        "Demand Response"
    )

    fig_demand = go.Figure()

    # Pure causal response
    fig_demand.add_trace(
        go.Scatter(
            x=(
                causal_price_changes
                * 100
            ),

            y=causal_demands,

            mode="lines",

            name="Pure Causal Response",

            line=dict(
                dash="dash"
            ),

            hovertemplate=(
                "Price change: %{x:.0f}%"
                "<br>Causal demand: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    # Optimized business response
    fig_demand.add_trace(
        go.Scatter(
            x=scenario_df[
                "price_change_pct"
            ],

            y=scenario_df[
                "expected_demand"
            ],

            mode="lines+markers",

            name="Optimized Business Response",

            hovertemplate=(
                "Price change: %{x:.0f}%"
                "<br>Expected demand: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    # AI optimum
    fig_demand.add_trace(
        go.Scatter(
            x=[optimum_change],

            y=[optimum_demand],

            mode="markers",

            marker=dict(
                size=20,
                symbol="star"
            ),

            name="AI Optimum",

            hovertemplate=(
                "★ AI Optimum"
                "<br>Price change: "
                f"{optimum_change:.1f}%"
                "<br>Demand: "
                f"{optimum_demand:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig_demand.update_layout(
        xaxis_title=(
            "Price Intervention (%)"
        ),

        yaxis_title=(
            "Expected Demand"
        ),

        height=470,

        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
    )

    st.plotly_chart(
        fig_demand,
        width="stretch"
    )

    st.caption(
        "Dashed line = pure causal price-response "
        "estimate. Solid line = expected demand after "
        "optimizing the marketing allocation for each "
        "price intervention. ★ = AI-selected decision."
    )

    # =========================================================
    # 2. PROFIT RESPONSE
    # =========================================================

    st.subheader(
        "Expected Profit Response"
    )

    fig_profit = go.Figure()

    fig_profit.add_trace(
        go.Scatter(
            x=scenario_df[
                "price_change_pct"
            ],

            y=scenario_df[
                "expected_profit"
            ],

            mode="lines+markers",

            name="Expected Profit",

            hovertemplate=(
                "Price change: %{x:.0f}%"
                "<br>Expected profit: ₹%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig_profit.add_trace(
        go.Scatter(
            x=[optimum_change],

            y=[optimum_profit],

            mode="markers",

            marker=dict(
                size=20,
                symbol="star"
            ),

            name="AI Optimum",

            hovertemplate=(
                "★ AI Optimum"
                "<br>Price change: "
                f"{optimum_change:.1f}%"
                "<br>Expected profit: ₹"
                f"{optimum_profit:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig_profit.update_layout(
        xaxis_title=(
            "Price Intervention (%)"
        ),

        yaxis_title=(
            "Expected Profit (₹)"
        ),

        height=470,

        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
    )

    st.plotly_chart(
        fig_profit,
        width="stretch"
    )

    # =========================================================
    # 3. AI SELECTED DECISION
    # =========================================================

    st.subheader(
        "AI-Selected Decision"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Recommended Price",
            f"₹{optimum_price:,.0f}"
        )

    with c2:

        st.metric(
            "Price Change",
            f"{optimum_change:.1f}%"
        )

    with c3:

        st.metric(
            "Expected Demand",
            f"{optimum_demand:,.0f}"
        )

    with c4:

        st.metric(
            "Expected Profit",
            f"₹{optimum_profit:,.0f}"
        )

    # =========================================================
    # 4. DECISION LANDSCAPE
    # =========================================================

    st.subheader(
        "Decision Landscape"
    )

    display_df = scenario_df[
        [
            "price_change_pct",
            "price",
            "marketing_spend",
            "expected_demand",
            "expected_profit",
            "profit_p05",
            "probability_of_loss",
            "CVaR_95",
            "objective"
        ]
    ].copy()

    display_df.columns = [
        "Price Change",
        "Price",
        "Marketing",
        "Expected Demand",
        "Expected Profit",
        "5th Percentile Profit",
        "Probability of Loss",
        "Downside CVaR",
        "Risk-Adjusted Objective"
    ]

    display_df[
        "Price Change"
    ] = display_df[
        "Price Change"
    ].map(
        lambda x: f"{x:+.0f}%"
    )

    display_df[
        "Price"
    ] = display_df[
        "Price"
    ].map(
        lambda x: f"₹{x:,.0f}"
    )

    display_df[
        "Marketing"
    ] = display_df[
        "Marketing"
    ].map(
        lambda x: f"₹{x:,.0f}"
    )

    display_df[
        "Expected Demand"
    ] = display_df[
        "Expected Demand"
    ].map(
        lambda x: f"{x:,.0f}"
    )

    display_df[
        "Expected Profit"
    ] = display_df[
        "Expected Profit"
    ].map(
        lambda x: f"₹{x:,.0f}"
    )

    display_df[
        "5th Percentile Profit"
    ] = display_df[
        "5th Percentile Profit"
    ].map(
        lambda x: f"₹{x:,.0f}"
    )

    display_df[
        "Probability of Loss"
    ] = display_df[
        "Probability of Loss"
    ].map(
        lambda x: f"{x * 100:.2f}%"
    )

    display_df[
        "Downside CVaR"
    ] = display_df[
        "Downside CVaR"
    ].map(
        lambda x: f"₹{x:,.0f}"
    )

    display_df[
        "Risk-Adjusted Objective"
    ] = display_df[
        "Risk-Adjusted Objective"
    ].map(
        lambda x: f"₹{x:,.0f}"
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

else:

    st.warning(
        "Scenario optimization results "
        "are unavailable."
    )

st.divider()

# ============================================================
# RISK
# ============================================================
# ============================================================

# ============================================================
# 6. RISK ANALYSIS
# ============================================================

st.header(
    "6. Risk Analysis"
)

risk = result.get(
    "risk",
    {}
)

optimization_result = result.get(
    "optimization",
    {}
)

probability_loss = risk.get(
    "probability_of_loss",
    optimization_result.get(
        "probability_of_loss",
        0
    )
)

risk_level = risk.get(
    "risk_level",
    "LOW"
)

profit_p05 = optimization_result.get(
    "profit_p05",
    0
)

cvar = risk.get(
    "CVaR_95",
    optimization_result.get(
        "cvar_95",
        0
    )
)

r1, r2, r3, r4 = st.columns(4)

with r1:

    st.metric(
        "Probability of Loss",
        f"{probability_loss * 100:.2f}%"
    )

with r2:

    st.metric(
        "Risk Level",
        risk_level
    )

with r3:

    st.metric(
        "5th Percentile Profit",
        f"₹{profit_p05:,.0f}"
    )

with r4:

    st.metric(
        "Downside CVaR 95%",
        f"₹{cvar:,.0f}"
    )

if probability_loss < 0.05:

    st.success(
        "LOW RISK — The simulated decision "
        "shows a low probability of loss."
    )

elif probability_loss < 0.15:

    st.warning(
        "MEDIUM RISK — The decision has "
        "meaningful downside exposure."
    )

else:

    st.error(
        "HIGH RISK — The decision has "
        "significant downside exposure."
    )

st.caption(
    "Risk estimates are based on Monte Carlo simulation "
    "of the recommended decision and should be interpreted "
    "as decision-support evidence rather than guarantees."
)


# ============================================================
# 7. AI DECISION AUDIT
# ============================================================

st.header(
    "7. AI Decision Audit"
)

audit = result.get(
    "audit",
    {}
)

audit_score = audit.get(
    "audit_score",
    0
)

confidence = audit.get(
    "confidence",
    "UNKNOWN"
)

approved = audit.get(
    "approved",
    False
)

a1, a2, a3 = st.columns(3)

with a1:

    st.metric(
        "Audit Score",
        f"{audit_score}/100"
    )

with a2:

    st.metric(
        "Confidence",
        confidence
    )

with a3:

    if approved:

        st.success(
            "Recommendation Approved"
        )

    else:

        st.error(
            "Recommendation Not Approved"
        )

warnings = audit.get(
    "warnings",
    []
)

if warnings:

    st.subheader(
        "Audit Warnings"
    )

    for warning in warnings:

        st.warning(
            warning
        )

else:

    st.info(
        "No material audit warnings were detected."
    )

st.caption(
    "The audit evaluates consistency across causal "
    "evidence, forecasting evidence, optimization, "
    "and risk analysis."
)


# ============================================================
# 8. AI DECISION COPILOT
# ============================================================

st.header(
    "8. AI Decision Copilot"
)

recommended_price = optimization_result.get(
    "price",
    0
)

recommended_change = (
    optimization_result.get(
        "price_change",
        0
    ) * 100
)

recommended_marketing = optimization_result.get(
    "marketing_spend",
    0
)

recommended_demand = optimization_result.get(
    "expected_demand",
    0
)

recommended_profit = optimization_result.get(
    "expected_profit",
    0
)

elasticity = optimization_result.get(
    "elasticity",
    causal_result.get(
        "estimated_elasticity",
        0
    )
)

st.markdown(
    f"""
### Recommendation

**Reduce the price to ₹{recommended_price:,.0f}**
corresponding to a **{recommended_change:.1f}% price change**,
while allocating **₹{recommended_marketing:,.0f}** to marketing.

### Why?

- The estimated causal price elasticity is
  **{elasticity:.3f}**.
- The optimizer estimates expected demand of
  **{recommended_demand:,.0f}**.
- Expected profit is approximately
  **₹{recommended_profit:,.0f}**.
- The decision has a simulated loss probability of
  **{probability_loss * 100:.2f}%**.
- The recommendation received an audit score of
  **{audit_score}/100** with **{confidence}** confidence.

### Decision interpretation

The causal model estimates how demand responds to price.
The optimizer then searches across price and marketing
decisions to identify a potentially better business outcome.

The recommendation is **decision support**, not autonomous
execution. Human approval remains required before applying
the decision in a real business environment.
"""
)

st.divider()

st.caption(
    "CAUSAL-X | AI-powered causal decision intelligence | "
    "Decision support, not autonomous execution."
)
