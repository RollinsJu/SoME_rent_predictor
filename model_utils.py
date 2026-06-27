"""Shared modeling helpers for the Streamlit rental prediction app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
TEST_SIZE = 0.20

KEYWORD_MAP = {
    "waterfront_flag": r"waterfront|water front|ocean view|oceanfront|beach|beachfront|harbor|water view|waterview|lakefront|riverfront|coastal|dock",
    "pet_friendly_flag": r"pet friendly|pets allowed|cats ok|dogs ok|pets welcome",
    "parking_flag": r"parking|garage|driveway|carport",
    "kw_pet": r"\bpet|\bdog|\bcat",
    "kw_parking": r"parking|garage|driveway|carport",
    "kw_laundry": r"laundry|washer|dryer",
    "kw_utilities": r"utilities included|heat included|hot water included|electric included",
    "kw_luxury": r"luxury|renovated|newly renovated|updated",
    "kw_house": r"\bhouse\b|single family|single-family",
    "kw_apartment": r"apartment|apt\b|unit",
}

CANDIDATE_NUMERIC_FEATURES = [
    "bedrooms_clean",
    "bathrooms_clean",
    "sqft_clean",
    "is_furnished_final",
    "waterfront_flag",
    "pet_friendly_flag",
    "parking_flag",
    "town_median_total_value",
    "town_mean_total_value",
    "town_median_land_value",
    "town_median_building_value",
    "town_median_building_size",
    "town_median_lot_size",
    "assessor_records",
    "search_location_mismatch_flag",
    "kw_pet",
    "kw_parking",
    "kw_laundry",
    "kw_utilities",
    "kw_luxury",
    "kw_house",
    "kw_apartment",
]

CANDIDATE_CATEGORICAL_FEATURES = ["source", "model_town_feature", "model_county_feature"]

FEATURE_LABELS = {
    "bedrooms_clean": "Bedrooms",
    "bathrooms_clean": "Bathrooms",
    "sqft_clean": "Square footage",
    "is_furnished_final": "Furnished listing",
    "waterfront_flag": "Waterfront/coastal wording",
    "pet_friendly_flag": "Pet-friendly wording",
    "parking_flag": "Parking/garage wording",
    "town_median_total_value": "Town median assessed value",
    "town_mean_total_value": "Town mean assessed value",
    "town_median_land_value": "Town median land value",
    "town_median_building_value": "Town median building value",
    "town_median_building_size": "Town median building size",
    "town_median_lot_size": "Town median lot size",
    "assessor_records": "Assessor record count",
    "search_location_mismatch_flag": "Search/location mismatch flag",
    "kw_pet": "Pet keyword",
    "kw_parking": "Parking keyword",
    "kw_laundry": "Laundry keyword",
    "kw_utilities": "Utilities included keyword",
    "kw_luxury": "Luxury/renovated keyword",
    "kw_house": "House wording",
    "kw_apartment": "Apartment/unit wording",
}


def candidate_data_paths(app_dir: str | Path) -> List[Path]:
    """Return likely locations for the joined v2 modeling dataset.

    The preferred class-project layout is:
        parent_folder/
            SoME_outputs/
                rental_assessor_town_join_latest.csv
            some_rent_streamlit_app/
                app.py

    The app also checks a few common SoME_outputs subfolders and finally falls
    back to the copy of the CSV packaged inside this app folder.
    """
    app_dir = Path(app_dir).resolve()
    parent_dir = app_dir.parent
    filename = "rental_assessor_town_join_latest.csv"

    return [
        parent_dir / "SoME_outputs" / filename,
        parent_dir / "SoME_outputs" / "data" / filename,
        parent_dir / "SoME_outputs" / "data" / "pipeline" / filename,
        parent_dir / "SoME_outputs" / "pipeline" / filename,
        parent_dir / "SoME_outputs" / "model_outputs" / "model_data" / filename,
        app_dir / filename,
    ]


def find_default_data_path(app_dir: str | Path) -> Path:
    """Find the first available v2 joined modeling dataset."""
    for path in candidate_data_paths(app_dir):
        if path.exists():
            return path

    checked = "\n".join(f"- {path}" for path in candidate_data_paths(app_dir))
    raise FileNotFoundError(
        "Could not find rental_assessor_town_join_latest.csv. Checked:\n" + checked
    )


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Load the v2 joined rental dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")
    return pd.read_csv(data_path)


def make_modeling_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], pd.Series]:
    """Create the core modeling features used in the SoME_5 workflow."""
    df = df_raw.copy()

    if "baseline_exclusion_flag" in df.columns:
        flag = df["baseline_exclusion_flag"].fillna(False).astype(bool)
        df = df.loc[~flag].copy()

    if "rent_price" not in df.columns:
        raise KeyError("rent_price is required to train this app")

    df["rent_price"] = pd.to_numeric(df["rent_price"], errors="coerce")
    df = df.loc[df["rent_price"].notna() & (df["rent_price"] > 0)].copy()

    for column in ["bedrooms", "bathrooms", "sqft", "is_furnished_final"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "bedrooms" in df.columns:
        df["bedrooms_clean"] = df["bedrooms"].where(df["bedrooms"].between(0, 10))
    else:
        df["bedrooms_clean"] = np.nan

    if "bathrooms" in df.columns:
        df["bathrooms_clean"] = df["bathrooms"].where(df["bathrooms"].between(0, 10))
    else:
        df["bathrooms_clean"] = np.nan

    if "sqft" in df.columns:
        df["sqft_clean"] = df["sqft"].where(df["sqft"].between(100, 6000))
    else:
        df["sqft_clean"] = np.nan

    if "is_furnished_final" not in df.columns:
        df["is_furnished_final"] = 0

    text_cols = [c for c in ["title", "combined_text", "raw_card_text", "location_raw"] if c in df.columns]
    if text_cols:
        text_series = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    else:
        text_series = pd.Series("", index=df.index)

    for new_col, pattern in KEYWORD_MAP.items():
        df[new_col] = text_series.str.contains(pattern, regex=True, na=False).astype(int)

    town_candidates = [c for c in ["modeling_town", "listing_town", "actual_town", "rental_town", "search_town", "search_query"] if c in df.columns]
    county_candidates = [c for c in ["modeling_county", "listing_county", "actual_county", "county_query"] if c in df.columns]

    if town_candidates:
        df["model_town_feature"] = df[town_candidates].bfill(axis=1).iloc[:, 0].astype(str).str.strip()
    else:
        df["model_town_feature"] = "unknown"

    if county_candidates:
        df["model_county_feature"] = df[county_candidates].bfill(axis=1).iloc[:, 0].astype(str).str.strip()
    else:
        df["model_county_feature"] = "unknown"

    numeric_features = [c for c in CANDIDATE_NUMERIC_FEATURES if c in df.columns]
    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    categorical_features = [c for c in CANDIDATE_CATEGORICAL_FEATURES if c in df.columns]
    for column in categorical_features:
        df[column] = df[column].fillna("unknown").astype(str)

    model_cols = numeric_features + categorical_features
    model_df = df[["rent_price"] + model_cols].copy()
    y_class = (model_df["rent_price"] > model_df["rent_price"].median()).astype(int)

    return model_df, numeric_features, categorical_features, y_class


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Build preprocessing used by tree-style models."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]), categorical_features),
    ])


def make_regression_model():
    """Use XGBoost when available, otherwise use a strong tree-based fallback."""
    if XGBOOST_AVAILABLE:
        return XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=3,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ), "XGBoost regression"

    return RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ), "Random Forest regression fallback"


def make_classification_model():
    """Use XGBoost when available, otherwise use a strong tree-based fallback."""
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ), "XGBoost classification"

    return RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ), "Random Forest classification fallback"


def train_models_from_data(df_raw: pd.DataFrame) -> Dict[str, object]:
    """Train the regression and classification models used by the app."""
    model_df, numeric_features, categorical_features, y_class = make_modeling_dataframe(df_raw)
    model_cols = numeric_features + categorical_features

    X = model_df[model_cols].copy()
    y = model_df["rent_price"].copy()
    threshold = float(y.median())

    X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
        X,
        y,
        y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )

    reg_model, reg_name = make_regression_model()
    cls_model, cls_name = make_classification_model()

    reg_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", reg_model),
    ])
    cls_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", cls_model),
    ])

    reg_pipeline.fit(X_train, y_train)
    cls_pipeline.fit(X_train, yc_train)

    reg_pred = reg_pipeline.predict(X_test)
    cls_proba = cls_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_mae": float(mean_absolute_error(y_test, reg_pred)),
        "test_r2": float(r2_score(y_test, reg_pred)),
        "test_roc_auc": float(roc_auc_score(yc_test, cls_proba)),
        "rent_threshold": threshold,
        "n_rows": int(len(model_df)),
        "n_features": int(len(model_cols)),
        "regression_model": reg_name,
        "classification_model": cls_name,
        "median_rent": float(y.median()),
        "mean_rent": float(y.mean()),
    }

    return {
        "df_raw": df_raw,
        "model_df": model_df,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_cols": model_cols,
        "reg_pipeline": reg_pipeline,
        "cls_pipeline": cls_pipeline,
        "metrics": metrics,
    }


def town_lookup_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create default town-level values for user controls."""
    df = df_raw.copy()
    town_col = "modeling_town" if "modeling_town" in df.columns else "search_query"
    county_col = "modeling_county" if "modeling_county" in df.columns else "county_query"

    numeric_defaults = [
        "town_median_total_value",
        "town_mean_total_value",
        "town_median_land_value",
        "town_median_building_value",
        "town_median_building_size",
        "town_median_lot_size",
        "assessor_records",
    ]
    keep = [town_col, county_col] + [c for c in numeric_defaults if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={town_col: "town", county_col: "county"})
    out["town"] = out["town"].fillna("Unknown").astype(str)
    out["county"] = out["county"].fillna("Unknown").astype(str)

    agg_dict = {"county": lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else "Unknown"}
    for col in numeric_defaults:
        if col in out.columns:
            agg_dict[col] = "median"

    return out.groupby("town", as_index=False).agg(agg_dict).sort_values("town")


def make_user_row(
    town_defaults: pd.Series,
    source: str,
    bedrooms: float,
    bathrooms: float,
    sqft: float | None,
    furnished: bool,
    waterfront: bool,
    pet_friendly: bool,
    parking: bool,
    laundry: bool,
    utilities: bool,
    luxury: bool,
    property_type: str,
    model_cols: List[str],
) -> pd.DataFrame:
    """Create one model-ready row from app form values."""
    row = {
        "bedrooms_clean": bedrooms,
        "bathrooms_clean": bathrooms,
        "sqft_clean": np.nan if sqft is None else sqft,
        "is_furnished_final": int(furnished),
        "waterfront_flag": int(waterfront),
        "pet_friendly_flag": int(pet_friendly),
        "parking_flag": int(parking),
        "kw_pet": int(pet_friendly),
        "kw_parking": int(parking),
        "kw_laundry": int(laundry),
        "kw_utilities": int(utilities),
        "kw_luxury": int(luxury),
        "kw_house": int(property_type == "House / single-family"),
        "kw_apartment": int(property_type == "Apartment / unit"),
        "search_location_mismatch_flag": 0,
        "source": source,
        "model_town_feature": town_defaults["town"],
        "model_county_feature": town_defaults["county"],
    }

    for col in [
        "town_median_total_value",
        "town_mean_total_value",
        "town_median_land_value",
        "town_median_building_value",
        "town_median_building_size",
        "town_median_lot_size",
        "assessor_records",
    ]:
        if col in town_defaults.index:
            row[col] = town_defaults[col]

    return pd.DataFrame([{col: row.get(col, np.nan) for col in model_cols}])


def get_feature_names(pipeline: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Return readable feature names after preprocessing."""
    pre = pipeline.named_steps["preprocessor"]
    names = list(numeric_features)
    if categorical_features:
        try:
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            names.extend(list(ohe.get_feature_names_out(categorical_features)))
        except Exception:
            names.extend(categorical_features)
    return names


def clean_feature_label(feature: str) -> str:
    """Make model feature names easier to read in the Streamlit charts."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("source_"):
        return "Listing source: " + feature.replace("source_", "").replace("_", " ").title()
    if feature.startswith("model_town_feature_"):
        return "Town: " + feature.replace("model_town_feature_", "").replace("_", " ").title()
    if feature.startswith("model_county_feature_"):
        return "County: " + feature.replace("model_county_feature_", "").replace("_", " ").title()
    return feature.replace("_", " ").title()


def top_model_features(bundle: Dict[str, object], top_n: int = 10) -> pd.DataFrame:
    """Return built-in model feature importance if available."""
    pipeline = bundle["reg_pipeline"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "feature_label", "importance"])

    names = get_feature_names(pipeline, bundle["numeric_features"], bundle["categorical_features"])
    values = np.asarray(model.feature_importances_)
    n = min(len(names), len(values))
    out = pd.DataFrame({"feature": names[:n], "importance": values[:n]})
    out = out.sort_values("importance", ascending=False).head(top_n).copy()
    out["feature_label"] = out["feature"].map(clean_feature_label)
    return out
