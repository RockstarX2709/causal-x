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
    use_container_width=True
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
    use_container_width=True
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


# ============================================================
# SCENARIO EXPLORER
# ============================================================

st.header(
    "5. Scenario Explorer"
)

st.markdown(
    """
Compare alternative price interventions and see how the
causal model changes demand and how the decision engine
changes expected profit.
"""
)

decisions = optimization.get(
    "all_decisions"
)

if decisions is not None:

    scenario_df = decisions.copy()

    # Convert price change to percentage
    scenario_df[
        "price_change_pct"
    ] = (
        scenario_df[
            "price_change"
        ] * 100
    )

    # Keep the best marketing decision for
    # each price intervention.
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
    )

    scenario_df = (
        scenario_df
        .sort_values(
            "price_change_pct"
        )
    )

    # ---------------------------------------------------------
    # DEMAND CHART
    # ---------------------------------------------------------

    fig_demand = go.Figure()

    fig_demand.add_trace(
        go.Scatter(
            x=scenario_df[
                "price_change_pct"
            ],

            y=scenario_df[
                "expected_demand"
            ],

            mode="lines+markers",

            name="Expected Demand",
        )
    )

    best_price_change = (
        optimization[
            "price_change"
        ] * 100
    )

    best_row = (
        scenario_df[
            np.isclose(
                scenario_df[
                    "price_change_pct"
                ],
                best_price_change
            )
        ]
    )

    if not best_row.empty:

        fig_demand.add_trace(
            go.Scatter(
                x=best_row[
                    "price_change_pct"
                ],

                y=best_row[
                    "expected_demand"
                ],

                mode="markers",

                marker=dict(
                    size=16,
                    symbol="star"
                ),

                name="AI Optimum",
            )
        )

    fig_demand.update_layout(
        title="Price Intervention → Expected Demand",

        xaxis_title=(
            "Price Change (%)"
        ),

        yaxis_title=(
            "Expected Demand"
        ),

        height=450,
    )

    st.plotly_chart(
        fig_demand,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # PROFIT CHART
    # ---------------------------------------------------------

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
        )
    )

    if not best_row.empty:

        fig_profit.add_trace(
            go.Scatter(
                x=best_row[
                    "price_change_pct"
                ],

                y=best_row[
                    "expected_profit"
                ],

                mode="markers",

                marker=dict(
                    size=16,
                    symbol="star"
                ),

                name="AI Optimum",
            )
        )

    fig_profit.update_layout(
        title="Price Intervention → Expected Profit",

        xaxis_title=(
            "Price Change (%)"
        ),

        yaxis_title=(
            "Expected Profit (₹)"
        ),

        height=450,
    )

    st.plotly_chart(
        fig_profit,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # DECISION TABLE
    # ---------------------------------------------------------

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
            "probability_of_loss",
            "CVaR_95",
        ]
    ].copy()

    display_df.columns = [
        "Price Change %",
        "Price",
        "Marketing",
        "Expected Demand",
        "Expected Profit",
        "Probability of Loss",
        "Downside CVaR",
    ]

    display_df[
        "Probability of Loss"
    ] = (
        display_df[
            "Probability of Loss"
        ] * 100
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Scenario data is unavailable."
    )


st.divider()


# ============================================================
# RISK
# ============================================================
# ============================================================
# SCENARIO EXPLORER
# ============================================================

st.header(
    "5. Scenario Explorer"
)

st.markdown(
    """
Explore how different price interventions affect demand,
profit, and risk. The ★ marks the decision selected by
the risk-adjusted optimizer.
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
    # PRICE CHANGE AS PERCENTAGE
    # ---------------------------------------------------------

    scenario_df[
        "price_change_pct"
    ] = (
        scenario_df[
            "price_change"
        ] * 100
    )

    # ---------------------------------------------------------
    # BEST DECISION AT EACH PRICE LEVEL
    #
    # There are multiple marketing budgets for each
    # price intervention. Keep the one with the highest
    # risk-adjusted objective.
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
    )

    scenario_df = (
        scenario_df
        .sort_values(
            "price_change_pct"
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # AI OPTIMUM
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
    # DEMAND RESPONSE
    # =========================================================

    st.subheader(
        "Causal Demand Response"
    )

    fig_demand = go.Figure()

    fig_demand.add_trace(
        go.Scatter(
            x=scenario_df[
                "price_change_pct"
            ],

            y=scenario_df[
                "expected_demand"
            ],

            mode="lines+markers",

            name="Expected Demand",

            hovertemplate=(
                "Price change: %{x:.0f}%"
                "<br>Expected demand: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    # AI optimum marker

    fig_demand.add_trace(
        go.Scatter(
            x=[optimum_change],

            y=[optimum_demand],

            mode="markers",

            marker=dict(
                size=18,
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

        height=430,

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
        use_container_width=True
    )

    # =========================================================
    # PROFIT RESPONSE
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
                size=18,
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

        height=430,

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
        use_container_width=True
    )

    # =========================================================
    # OPTIMAL DECISION
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
    # DECISION LANDSCAPE
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
            "probability_of_loss",
            "CVaR_95"
        ]
    ].copy()

    display_df.columns = [
        "Price Change",
        "Price",
        "Marketing",
        "Expected Demand",
        "Expected Profit",
        "Probability of Loss",
        "Downside CVaR"
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

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Scenario optimization results "
        "are unavailable."
    )


st.divider()