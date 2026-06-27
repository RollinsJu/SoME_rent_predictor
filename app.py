"""
Southern Maine Rental Price Prediction App

This Streamlit app uses the final v2 joined rental/assessor dataset from the class project.
It trains a rent regression model and an above-median-rent classifier when the app starts,
then lets a user enter interpretable rental features to generate predictions.
"""

from __future__ import annotations

from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (
    find_default_data_path,
    load_raw_data,
    make_user_row,
    top_model_features,
    town_lookup_table,
    train_models_from_data,
)

# -----------------------------
# Page setup and styling
# -----------------------------

st.set_page_config(
    page_title="Southern Maine Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background: linear-gradient(180deg, #F7F8FC 0%, #FFFFFF 45%); }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, #2F4B7C 0%, #665191 55%, #00A6A6 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(47, 75, 124, 0.22);
        margin-bottom: 1rem;
    }
    .hero-card h1 { margin-bottom: 0.25rem; font-size: 2.2rem; }
    .hero-card p { font-size: 1.02rem; margin-bottom: 0; opacity: 0.94; }
    .metric-card {
        padding: 1rem;
        border-radius: 1rem;
        background: white;
        border: 1px solid #E6E8F0;
        box-shadow: 0 4px 16px rgba(25, 36, 65, 0.06);
    }
    .small-note { color: #5D6475; font-size: 0.88rem; }
    .prediction-box {
        padding: 1.2rem;
        border-radius: 1rem;
        background: #F0F7FF;
        border: 1px solid #CFE3FF;
        margin-top: 0.5rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 1rem;
        background: #FFF8E8;
        border: 1px solid #F4D58D;
        color: #614A00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Shared project configuration
# -----------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = find_default_data_path(APP_DIR)


def money(value: float, decimals: int = 0) -> str:
    """Format dollars consistently across the app."""
    if pd.isna(value):
        return "Not available"
    return f"${value:,.{decimals}f}"


@st.cache_data(show_spinner=False)
def cached_raw_data() -> pd.DataFrame:
    """Load the shared v2 joined rental dataset."""
    return load_raw_data(DATA_PATH)


@st.cache_resource(show_spinner=True)
def train_models():
    """Train and cache the regression and classification models used by the app."""
    return train_models_from_data(cached_raw_data())


def readable_bar_chart(df: pd.DataFrame, category: str, value: str, title: str, value_prefix: str = "$"):
    """Create a horizontal bar chart so long category labels stay readable."""
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{value}:Q", title=None, axis=alt.Axis(format="$,.0f" if value_prefix == "$" else ",.0f")),
            y=alt.Y(f"{category}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=360)),
            tooltip=[alt.Tooltip(f"{category}:N", title="Category"), alt.Tooltip(f"{value}:Q", title=value, format=",.0f")],
        )
        .properties(title=title, height=max(160, 36 * len(df)))
    )
    return chart


def get_town_row(town_name: str) -> pd.Series:
    """Return the lookup row for a selected town."""
    return town_defaults_df.loc[town_defaults_df["town"] == town_name].iloc[0]


def predict_for(town_name: str, source_name: str, bedrooms_value: int, bathrooms_value: float, sqft_value: int) -> tuple[float, float]:
    """Create a single prediction for a scenario shown in the app."""
    scenario_row = make_user_row(
        town_defaults=get_town_row(town_name),
        source=source_name,
        bedrooms=bedrooms_value,
        bathrooms=bathrooms_value,
        sqft=sqft_value,
        furnished=furnished,
        waterfront=waterfront,
        pet_friendly=pet_friendly,
        parking=parking,
        laundry=laundry,
        utilities=utilities,
        luxury=luxury,
        property_type=property_type,
        model_cols=bundle["model_cols"],
    )
    rent = float(bundle["reg_pipeline"].predict(scenario_row)[0])
    prob = float(bundle["cls_pipeline"].predict_proba(scenario_row)[0, 1])
    return rent, prob


# -----------------------------
# App content
# -----------------------------

bundle = train_models()
metrics = bundle["metrics"]
town_defaults_df = town_lookup_table(bundle["df_raw"])

st.markdown(
    """
    <div class="hero-card">
        <h1>Southern Maine Rental Price Predictor</h1>
        <p>Estimate monthly rent from public listing details and town-level assessor context, then classify whether the listing is likely above the project median rent.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_df = bundle["model_df"]
town_options = town_defaults_df["town"].dropna().astype(str).tolist()
default_town = "Portland" if "Portland" in town_options else town_options[0]
source_options = sorted(model_df["source"].dropna().astype(str).unique()) if "source" in model_df.columns else ["craigslist"]

sqft_series = pd.to_numeric(model_df.get("sqft_clean", pd.Series(dtype=float)), errors="coerce")
sqft_median = sqft_series.median()
sqft_q95 = sqft_series.quantile(0.95)
default_sqft = 900 if pd.isna(sqft_median) or sqft_median < 100 else int(sqft_median)
sqft_min = 100
sqft_max = 3000 if pd.isna(sqft_q95) else int(max(3000, min(6000, sqft_q95)))

st.sidebar.title("Rental listing inputs")
st.sidebar.caption("These controls focus on interpretable features from the final v2 model and explainability work.")
st.sidebar.caption("Town and source are single-listing inputs. Use the comparison charts below to compare multiple towns or sources.")
st.sidebar.caption(f"Data file: `{DATA_PATH.name}`")

with st.sidebar.form("prediction_form"):
    town = st.selectbox(
        "Town (type to search)",
        town_options,
        index=town_options.index(default_town),
        help="Start typing to filter the list. Predictions use towns observed in the training data."
    )
    town_row = get_town_row(town)

    source = st.selectbox(
        "Listing source",
        source_options,
        help="A single listing comes from one source. The app includes a source-comparison chart below."
    )

    bedrooms = st.slider("Bedrooms", min_value=0, max_value=6, value=2, step=1)
    bathrooms = st.slider("Bathrooms", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    sqft_value = st.slider("Square footage", min_value=sqft_min, max_value=sqft_max, value=min(default_sqft, sqft_max), step=50)

    furnished = st.toggle("Furnished", value=False)

    st.markdown("**Listing text/features**")
    waterfront = st.toggle("Waterfront / coastal language", value=False)
    pet_friendly = st.toggle("Pet-friendly language", value=False)
    parking = st.toggle("Parking / garage language", value=False)
    laundry = st.toggle("Laundry language", value=False)
    utilities = st.toggle("Utilities included language", value=False)
    luxury = st.toggle("Luxury / renovated language", value=False)
    property_type = st.radio("Property wording", ["Apartment / unit", "House / single-family", "Other / unclear"], horizontal=False)

    submitted = st.form_submit_button("Predict rent", use_container_width=True)

user_row = make_user_row(
    town_defaults=town_row,
    source=source,
    bedrooms=bedrooms,
    bathrooms=bathrooms,
    sqft=sqft_value,
    furnished=furnished,
    waterfront=waterfront,
    pet_friendly=pet_friendly,
    parking=parking,
    laundry=laundry,
    utilities=utilities,
    luxury=luxury,
    property_type=property_type,
    model_cols=bundle["model_cols"],
)

predicted_rent = float(bundle["reg_pipeline"].predict(user_row)[0])
above_median_prob = float(bundle["cls_pipeline"].predict_proba(user_row)[0, 1])
class_label = "Above median" if above_median_prob >= 0.50 else "At or below median"
town_observed = model_df.loc[model_df.get("model_town_feature", pd.Series(index=model_df.index)).astype(str) == str(town), "rent_price"]
town_median = float(town_observed.median()) if len(town_observed.dropna()) else np.nan

main_col, side_col = st.columns([1.55, 1.0], gap="large")

with main_col:
    st.subheader("Prediction")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Predicted monthly rent", money(predicted_rent))
    metric_cols[1].metric("Median threshold", money(metrics["rent_threshold"]))
    metric_cols[2].metric("Above-median probability", f"{above_median_prob:.1%}")

    st.markdown(
        f"""
        <div class="prediction-box">
        <strong>Classification result:</strong> {class_label}<br>
        <span class="small-note">The classifier estimates whether this listing is above the project median rent, not whether the rent is objectively fair.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_df = pd.DataFrame({
        "Comparison": ["Predicted listing", "Project median", f"{town} observed median"],
        "Monthly rent": [predicted_rent, metrics["median_rent"], town_median],
    })
    st.altair_chart(readable_bar_chart(compare_df, "Comparison", "Monthly rent", "Predicted rent compared with dataset benchmarks"), use_container_width=True)

    st.caption(
        "The town comparison is based on observed listings in the v2 joined dataset. Sparse towns may have less stable medians."
    )

with side_col:
    st.subheader("Model health check")
    st.markdown(
        f"""
        <div class="metric-card">
        <strong>Rows used:</strong> {metrics['n_rows']:,}<br>
        <strong>Input features before one-hot encoding:</strong> {metrics['n_features']}<br>
        <strong>Regression model:</strong> {metrics['regression_model']}<br>
        <strong>Classifier:</strong> {metrics['classification_model']}<br>
        <strong>Holdout MAE:</strong> {money(metrics['test_mae'])}<br>
        <strong>Holdout R2:</strong> {metrics['test_r2']:.3f}<br>
        <strong>Classifier ROC-AUC:</strong> {metrics['test_roc_auc']:.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="warning-box">
        <strong>Class project note:</strong> This app demonstrates the modeling workflow. It should be treated as a prototype, not a production pricing tool.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["What drives predictions", "Scenario comparisons", "Dataset context", "How to use this prototype"])

with tab1:
    st.subheader("Important model features")
    feature_df = top_model_features(bundle, top_n=12)
    if feature_df.empty:
        st.info("The selected model does not expose built-in feature importances. Check the SoME_5 SHAP outputs for full explainability.")
    else:
        plot_features = feature_df.copy()
        plot_features["importance_pct"] = 100 * plot_features["importance"] / plot_features["importance"].sum()
        chart = (
            alt.Chart(plot_features)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("importance_pct:Q", title="Share of shown model importance (%)"),
                y=alt.Y("feature_label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=380)),
                tooltip=[alt.Tooltip("feature_label:N", title="Feature"), alt.Tooltip("importance_pct:Q", title="Importance share", format=".1f")],
            )
            .properties(title="Global model feature importance (largest to smallest)", height=430)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "This is a global model-importance chart, so it does not change when the input scenario changes. It shows which features the trained model used most overall."
        )

    st.markdown("**How to interpret the controls**")
    st.write(
        "Bedrooms, bathrooms, square footage, town, source, furnished status, and listing-text flags are treated as model signals. "
        "The assessor fields are filled automatically from the selected town because this prototype uses town-level assessor context."
    )

with tab2:
    st.subheader("Scenario comparisons")

    st.markdown("**Bedroom sensitivity**")
    bedroom_rows = []
    for b in range(0, 7):
        rent, prob = predict_for(town, source, b, bathrooms, sqft_value)
        bedroom_rows.append({"Bedrooms": b, "Predicted rent": rent, "Above-median probability": prob})
    bedroom_df = pd.DataFrame(bedroom_rows)
    line_chart = (
        alt.Chart(bedroom_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Bedrooms:O", title="Bedrooms"),
            y=alt.Y("Predicted rent:Q", title="Predicted monthly rent", axis=alt.Axis(format="$,.0f")),
            tooltip=["Bedrooms:O", alt.Tooltip("Predicted rent:Q", format="$,.0f"), alt.Tooltip("Above-median probability:Q", format=".1%")],
        )
        .properties(title="Predicted rent as bedroom count changes", height=280)
    )
    st.altair_chart(line_chart, use_container_width=True)

    compare_cols = st.columns(2)
    with compare_cols[0]:
        st.markdown("**Compare listing sources**")
        selected_sources = st.multiselect(
            "Sources to compare",
            source_options,
            default=source_options,
            help="This does not mean one listing has multiple sources. It compares the same listing assumptions as if they appeared on different sources."
        )
        if selected_sources:
            source_rows = []
            for s in selected_sources:
                rent, prob = predict_for(town, s, bedrooms, bathrooms, sqft_value)
                source_rows.append({"Source": s, "Predicted rent": rent, "Above-median probability": prob})
            source_df = pd.DataFrame(source_rows)
            st.altair_chart(readable_bar_chart(source_df, "Source", "Predicted rent", "Same listing assumptions by source"), use_container_width=True)

    with compare_cols[1]:
        st.markdown("**Compare towns**")
        common_towns = model_df["model_town_feature"].value_counts().head(8).index.astype(str).tolist() if "model_town_feature" in model_df.columns else town_options[:8]
        default_towns = [t for t in [town, "Portland", "South Portland", "Biddeford"] if t in town_options]
        if not default_towns:
            default_towns = common_towns[:3]
        selected_towns = st.multiselect(
            "Towns to compare",
            town_options,
            default=default_towns,
            help="This compares the same listing assumptions across different town contexts learned from the data."
        )
        if selected_towns:
            town_rows = []
            for t in selected_towns:
                rent, prob = predict_for(t, source, bedrooms, bathrooms, sqft_value)
                town_rows.append({"Town": t, "Predicted rent": rent, "Above-median probability": prob})
            town_compare_df = pd.DataFrame(town_rows)
            st.altair_chart(readable_bar_chart(town_compare_df, "Town", "Predicted rent", "Same listing assumptions by town"), use_container_width=True)

with tab3:
    st.subheader("Dataset context")
    source_counts = model_df["source"].value_counts().rename_axis("Source").reset_index(name="Listings") if "source" in model_df.columns else pd.DataFrame()
    if not source_counts.empty:
        st.altair_chart(readable_bar_chart(source_counts, "Source", "Listings", "Listings by source", value_prefix=""), use_container_width=True)

    if "bedrooms_clean" in model_df.columns:
        bedroom_summary = (
            model_df.dropna(subset=["bedrooms_clean"])
            .assign(Bedrooms=lambda d: d["bedrooms_clean"].round().astype(int))
            .query("Bedrooms >= 0 and Bedrooms <= 6")
            .groupby("Bedrooms", as_index=False)["rent_price"]
            .median()
            .rename(columns={"rent_price": "Median rent"})
        )
        bedroom_context_chart = (
            alt.Chart(bedroom_summary)
            .mark_line(point=True)
            .encode(
                x=alt.X("Bedrooms:O", title="Bedrooms"),
                y=alt.Y("Median rent:Q", title="Observed median rent", axis=alt.Axis(format="$,.0f")),
                tooltip=["Bedrooms:O", alt.Tooltip("Median rent:Q", format="$,.0f")],
            )
            .properties(title="Observed median rent by bedroom count", height=280)
        )
        st.altair_chart(bedroom_context_chart, use_container_width=True)

    st.write(
        f"The app trains from the v2 joined modeling file with {metrics['n_rows']:,} usable rental records. "
        "The full project pipeline began with scraped rental listings and then joined in town-level assessor/property context."
    )
    st.caption(f"Dataset loaded from: {DATA_PATH}")

    preview_cols = [c for c in ["source", "model_town_feature", "rent_price", "bedrooms_clean", "bathrooms_clean", "sqft_clean", "is_furnished_final"] if c in model_df.columns]
    st.dataframe(model_df[preview_cols].head(20), use_container_width=True)

with tab4:
    st.subheader("Use case framing")
    st.markdown(
        """
        - **Potential end users:** landlords comparing a planned rent against similar listings and tenants comparing a listing against model expectations.
        - **Prediction task:** estimate monthly rent as a continuous value.
        - **Classification task:** estimate whether the listing is above the project median rent.
        - **Important limitation:** this is a prototype built from public listings and town-level context, not a production pricing system.
        - **Best next step:** improve feature quality with exact square footage, utilities, lease terms, parking, pets, seasonality, and better address-level matching.
        """
    )
