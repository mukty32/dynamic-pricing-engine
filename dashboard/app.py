"""Phase 5: interactive pricing dashboard.

Start the API first, then the dashboard (two terminals, both in the project root):

    uvicorn pricing_engine.api.main:app
    streamlit run dashboard/app.py

Streamlit re-runs this whole script top to bottom every time you move a slider.
That's the entire programming model: widgets in, charts out.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# `streamlit run dashboard/app.py` only puts the dashboard/ folder on the import path;
# add the project root so `import dashboard.client` works however the app is started.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.analysis import explain_binding, fit_exponential, pen_and_paper_optimum  # noqa: E402
from dashboard.client import DEFAULT_API_URL, ApiError, PricingClient  # noqa: E402

API_URL = os.environ.get("PRICING_API_URL", DEFAULT_API_URL)
BLUE = "#2a78d6"  # the one data colour (each chart shows a single series)
MUTED = "#898781"  # reference lines and their labels: readable in light and dark mode

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="🚕", layout="wide")


# ----------------------------------------------------------------------------- data access
@st.cache_resource
def get_client(url: str) -> PricingClient:
    return PricingClient(url)


@st.cache_data(ttl=60, show_spinner=False)
def load_meta(url: str) -> tuple[dict, list[dict]]:
    client = get_client(url)
    return client.health(), client.zones()


@st.cache_data(ttl=300, show_spinner=False)
def load_curve(url: str, payload: dict) -> dict:
    # Cached by payload: dragging a slider back to an earlier value is instant.
    return get_client(url).demand_curve(payload)


# ----------------------------------------------------------------------------- charts
def x_axis(lower: float, upper: float) -> alt.X:
    return alt.X("price:Q", title="Our price ($)", scale=alt.Scale(domain=[lower, upper], nice=False))


def hover_layer(base: alt.Chart, y_field: str) -> alt.Chart:
    """Invisible, oversized dots that show a tooltip anywhere along the line."""
    return base.mark_circle(size=300, opacity=0).encode(
        y=f"{y_field}:Q",
        tooltip=[
            alt.Tooltip("price:Q", title="price", format="$.2f"),
            alt.Tooltip("expected_demand:Q", title="ride requests", format=".1f"),
            alt.Tooltip("expected_rides:Q", title="rides served", format=".1f"),
            alt.Tooltip("expected_revenue:Q", title="revenue", format="$,.0f"),
        ],
    )


def optimum_layers(opt: dict, y_field: str, x: alt.X) -> alt.LayerChart:
    point = pd.DataFrame([{"price": opt["price"], y_field: opt[y_field], "label": f"best: ${opt['price']:.2f}"}])
    chart = alt.Chart(point).encode(x=x, y=f"{y_field}:Q")
    return chart.mark_point(filled=True, size=160, color=BLUE, stroke="white", strokeWidth=2) + chart.mark_text(
        dy=-16, fontWeight="bold", color=MUTED
    ).encode(text="label:N")


def vertical_rule(value: float, label: str, x: alt.X) -> alt.LayerChart:
    data = pd.DataFrame([{"price": value, "label": label}])
    chart = alt.Chart(data).encode(x=x)
    return chart.mark_rule(strokeDash=[4, 4], color=MUTED, clip=True) + chart.mark_text(
        align="left", dx=4, y=8, color=MUTED, clip=True
    ).encode(text="label:N")


def revenue_chart(points: pd.DataFrame, opt: dict, base_fare: float) -> alt.LayerChart:
    x = x_axis(opt["lower_bound"], opt["upper_bound"])
    base = alt.Chart(points).encode(x=x)
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(
        y=alt.Y("expected_revenue:Q", title="Expected revenue per hour ($)")
    )
    return (
        line
        + hover_layer(base, "expected_revenue")
        + vertical_rule(base_fare, "base fare", x)
        + optimum_layers(opt, "expected_revenue", x)
    ).properties(height=340, title="Revenue = price × rides served")


def demand_chart(points: pd.DataFrame, opt: dict) -> alt.LayerChart:
    x = x_axis(opt["lower_bound"], opt["upper_bound"])
    base = alt.Chart(points).encode(x=x)
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(
        y=alt.Y("expected_demand:Q", title="Expected ride requests per hour")
    )
    capacity = pd.DataFrame([{"capacity": opt["capacity"], "label": f"driver capacity: {opt['capacity']:.0f} rides"}])
    cap = alt.Chart(capacity).encode(y="capacity:Q")
    cap_layers = cap.mark_rule(strokeDash=[4, 4], color=MUTED) + cap.mark_text(
        align="left", x=4, dy=-7, color=MUTED
    ).encode(text="label:N")
    return (
        line + hover_layer(base, "expected_demand") + cap_layers + optimum_layers(opt, "expected_demand", x)
    ).properties(height=340, title="Demand curve D(p)")


# ----------------------------------------------------------------------------- page
st.title("🚕 Dynamic Pricing Engine")
st.caption(
    "Move the sliders on the left. Every change asks the pricing API for a fresh "
    "demand forecast and the revenue-maximising price."
)

try:
    health, zones = load_meta(API_URL)
except ApiError as exc:
    st.error(f"**{exc}**\n\nStart the API in another terminal, then refresh this page:")
    st.code("uvicorn pricing_engine.api.main:app", language="bash")
    st.stop()

fares = {zone["name"]: zone["base_fare"] for zone in zones}

with st.sidebar:
    st.header("Market situation")
    zone = st.selectbox("Zone", list(fares), format_func=str.title, key="zone")
    fare = fares[zone]
    day = st.date_input(
        "Date", value=date(2025, 11, 18), min_value=date(2025, 1, 1), max_value=date(2026, 12, 31), key="day"
    )
    hour = st.slider("Hour of day", 0, 23, 18, format="%d:00", key="hour")
    is_raining = st.toggle("Raining", value=False, key="rain")

    st.header("Supply & demand")
    drivers = st.slider("Drivers online (supply)", 0, 300, 40, key="drivers")
    demand_multiplier = st.slider(
        "Demand shock (×)",
        0.5,
        3.0,
        1.0,
        step=0.1,
        key="shock",
        help="Scale the model's forecast, e.g. 2.0 when a concert ends nearby.",
    )
    competitor_price = st.slider(
        "Competitor price ($)",
        round(0.5 * fare, 1),
        round(4.0 * fare, 1),
        round(1.5 * fare, 1),
        step=0.5,
        key="competitor",
    )

    with st.expander("Business rules"):
        max_multiplier = st.slider("Surge cap (× base fare)", 1.0, 5.0, 3.0, step=0.1, key="cap")
        premium = st.slider("Max premium over competitor (%)", 0, 200, 50, step=5, key="premium")
        min_multiplier = st.slider("Price floor (× base fare)", 0.5, 1.0, 0.8, step=0.05, key="floor")

    st.divider()
    st.caption(f"Model: **{health['model_name']}** · API v{health['version']}  \n`{API_URL}`")

rules = {"min_multiplier": min_multiplier, "max_multiplier": max_multiplier, "max_competitor_premium": premium / 100}
payload = {
    "timestamp": f"{day.isoformat()}T{hour:02d}:00:00",
    "zone": zone,
    "is_raining": is_raining,
    "competitor_price": competitor_price,
    "drivers_available": drivers,
    "demand_multiplier": demand_multiplier,
    "rules": rules,
    "n_points": 150,
}

try:
    result = load_curve(API_URL, payload)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

opt = result["optimum"]
points = pd.DataFrame(result["points"])

# --- KPI row ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Optimal price", f"${opt['price']:.2f}", f"{opt['surge_multiplier']:.2f}× base fare", delta_color="off")
k2.metric("Revenue / hour", f"${opt['expected_revenue']:,.0f}", f"{opt['revenue_uplift']:+.0%} vs base fare")
k3.metric(
    "Rides served / hour", f"{opt['expected_rides']:.0f}", f"of {opt['capacity']:.0f} capacity", delta_color="off"
)
k4.metric("Riders who get a car", f"{opt['fulfillment_rate']:.0%}")

headline, explanation = explain_binding(
    opt["binding_constraint"],
    price=opt["price"],
    capacity=opt["capacity"],
    competitor_price=competitor_price,
    base_fare=fare,
    rules=rules,
)
st.info(f"**Why ${opt['price']:.2f}? {headline}.** {explanation}")

# --- Charts ---
left, right = st.columns(2)
left.altair_chart(revenue_chart(points, opt, fare), width="stretch")
right.altair_chart(demand_chart(points, opt), width="stretch")
st.caption(
    f"The x-axis spans the allowed price range ${opt['lower_bound']:.2f} to ${opt['upper_bound']:.2f}. "
    "Hover over a line for exact values."
)

# --- The math, live ---
with st.expander("🧮 The math behind this price (with today's numbers)"):
    beta, C = fit_exponential(points["price"], points["expected_demand"])
    steps = pen_and_paper_optimum(beta, C, opt["capacity"], opt["lower_bound"], opt["upper_bound"])
    st.markdown("**1. Read the demand curve.** Fitting a straight line to $\\ln D$ against $p$ recovers:")
    st.latex(rf"D(p) \approx {C:,.1f}\; e^{{-{beta:.4f}\,p}}")
    st.markdown("**2. Natural peak** (where $R'(p)=0$, ignoring supply):")
    st.latex(rf"p_{{\text{{peak}}}} = \frac{{1}}{{\beta}} = \frac{{1}}{{{beta:.4f}}} = \${steps['natural_peak']:.2f}")
    if steps["market_clearing"] is not None:
        st.markdown(f"**3. Market-clearing price** (demand = capacity $K = {opt['capacity']:.0f}$):")
        st.latex(
            rf"p_K = \frac{{\ln(C/K)}}{{\beta}} = \frac{{\ln({C:,.1f}/{opt['capacity']:.0f})}}{{{beta:.4f}}}"
            rf" = \${steps['market_clearing']:.2f}"
        )
    st.markdown("**4. Take the larger, then clip to the business rules:**")
    st.latex(
        rf"p^* = \operatorname{{clip}}\big(\max(p_{{\text{{peak}}}},\, p_K),\ "
        rf"{opt['lower_bound']:.2f},\ {opt['upper_bound']:.2f}\big) = \${steps['optimum']:.2f}"
    )
    st.success(f"Pen-and-paper answer **${steps['optimum']:.2f}** vs SciPy engine **${opt['price']:.2f}**")

with st.expander("Raw API response (for developers)"):
    st.json(opt)
