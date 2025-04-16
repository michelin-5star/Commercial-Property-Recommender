import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import zipfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

property_type_flags = ['Office', 'Retail', 'Industrial', 'Healthcare', 'Other',
                       'Land', 'Investment', 'Multifamily', 'Hospitality']



@st.cache_data
def load_data():
    # Read zipped lease CSV
    with zipfile.ZipFile("lease_data.csv.zip") as lease_zip:
        with lease_zip.open("lease_data.csv") as lease_file:
            lease_df = pd.read_csv(lease_file)

    # Read zipped sales CSV
    with zipfile.ZipFile("sales_data.csv.zip") as sales_zip:
        with sales_zip.open("sales_data.csv") as sales_file:
            sales_df = pd.read_csv(sales_file)

    return sales_df, lease_df

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_sbert_similarity(df):
    df["combined_features"] = df["CLEANED_DESCRIPTION"].fillna('') + " " + df["CLEANED_HIGHLIGHTS"].fillna('')
    embeddings = sbert_model.encode(df["combined_features"].tolist(), show_progress_bar=False)
    return cosine_similarity(embeddings)

def prioritize_description(row):
    return 1 if (str(row.get("DESCRIPTION", "")).strip() or str(row.get("HIGHLIGHTS", "")).strip()) else 0

def transit_preference(row, prefs):
    rng = prefs.get("preferred_transit_range")
    return int(row.get(f"has_transit_{rng}mi", False)) if rng else 0

def calculate_final_score(row, cosine_sim, idx, user_preferences, roi_column=None):
    cosine_score = cosine_sim[idx][idx]
    transit_score = row["transit_score"]
    description_priority = row["description_priority"]
    roi_score = row[roi_column] if roi_column and roi_column in row else 0

    if user_preferences["type"] == "sale":
        return 0.3 * cosine_score + 0.2 * transit_score + 0.4 * roi_score + 0.1 * description_priority
    elif user_preferences["type"] == "lease":
        return 0.6 * cosine_score + 0.4 * transit_score + 0.2 * description_priority
    else:
        return 0

def recommend_properties(user_preferences, sales_df, lease_df, top_n=10):
    if user_preferences["type"] == "sale":
        df = sales_df.copy()
        price_column = "PRICE_Modified"
        roi_column = "Normalized_ROI"
    else:
        df = lease_df.copy()
        price_column = "rate_per_sf_year"
        roi_column = None

    type_filters = user_preferences["property_types"]
    df = df[
        (df["STATE_ABBR"] == user_preferences["state_abbr"]) &
        (df[price_column] >= user_preferences["price_range"][0]) &
        (df[price_column] <= user_preferences["price_range"][1]) &
        (df["BUILDING_SF_Modified_Numeric_Final"] >= user_preferences["building_size"][0]) &
        (df["BUILDING_SF_Modified_Numeric_Final"] <= user_preferences["building_size"][1]) &
        (df[type_filters].any(axis=1))
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["transit_score"] = df.apply(lambda row: transit_preference(row, user_preferences), axis=1)
    df["description_priority"] = df.apply(prioritize_description, axis=1)
    cosine_sim = calculate_sbert_similarity(df)
    df["final_score"] = [calculate_final_score(df.iloc[i], cosine_sim, i, user_preferences, roi_column) for i in range(len(df))]

    return df.sort_values("final_score", ascending=False).head(top_n)

# === Streamlit UI ===
st.title("🏢 Commercial Property Recommender")

sales_df, lease_df = load_data()

st.sidebar.header("User Preferences")
prop_type = st.sidebar.selectbox("Property Type", ["sale", "lease"])
state = st.sidebar.text_input("State Abbreviation", value="TX")

if prop_type == "lease":
    min_price = st.sidebar.number_input("Min Lease Rate ($/SF/Year)", value=10.0, min_value=0.0, step=1.0)
    max_price = st.sidebar.number_input("Max Lease Rate ($/SF/Year)", value=60.0, min_value=0.0, step=1.0)
else:
    min_price = st.sidebar.number_input("Min Sale Price ($)", value=100000.0, min_value=0.0, step=50000.0)
    max_price = st.sidebar.number_input("Max Sale Price ($)", value=1000000.0, min_value=0.0, step=50000.0)

size_range = st.sidebar.slider("Building Size (SF)", 0, 10000, (500, 5000), step=100)
prop_types = st.sidebar.multiselect("Property Types", property_type_flags, default=["Office"])
transit_dist = st.sidebar.selectbox("Transit Distance Preference", [None, 1, 3, 5])

if st.sidebar.button("Get Recommendations"):
    user_prefs = {
        "type": prop_type,
        "state_abbr": state,
        "price_range": (min_price, max_price),
        "building_size": size_range,
        "property_types": prop_types,
        "preferred_transit_range": transit_dist
    }

    output = recommend_properties(user_prefs, sales_df, lease_df, top_n=10)

    if output.empty:
        st.warning("No properties matched your preferences.")
    else:
        st.success("Top Recommended Properties:")
        cols = ["ID", "ADDRESS", "CITY", "STATE_ABBR", "final_score", "BUILDING_SF_Modified_Numeric_Final"] + prop_types
        if prop_type == "lease":
            cols += ["PredLow", "PredHigh", "rate_per_sf_year", "LeaseDealScore"]
            price_col = "rate_per_sf_year"
            score_col = "LeaseDealScore"
            price_range = (0, 100)
        else:
            cols += ["PredLow", "PredHigh", "PRICE_Modified", "DealScore"]
            price_col = "PRICE_Modified"
            score_col = "DealScore"
            price_range = (0, 10000000)

        column_config = {
            price_col: st.column_config.NumberColumn(
                label="Price",
                min_value=price_range[0],
                max_value=price_range[1]
            ),
            score_col: st.column_config.NumberColumn(
                label="Deal Score",
                min_value=5,
                max_value=10
            ),
            "final_score": st.column_config.NumberColumn(
                label="Final Score",
                min_value=0.0,
                max_value=1.0
            )
        }

        st.dataframe(output[cols], column_config=column_config, use_container_width=True)

        # === Interactive Map with Tooltips ===
        st.subheader("🗺️ Property Locations")
        if "LATITUDE" in output.columns and "LONGITUDE" in output.columns:
            map_data = output.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
            map_data["tooltip"] = map_data["ADDRESS"] + ", " + map_data["CITY"] + " (" + map_data["DealScore"].round(2).astype(str) + ")"

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[LONGITUDE, LATITUDE]',
                get_radius=100,
                get_fill_color=[255, 0, 0],
                pickable=True,
                radius_min_pixels=5,
                radius_max_pixels=15
            )

            tooltip = {
                "html": "<b>{tooltip}</b>",
                "style": {
                    "backgroundColor": "white",
                    "color": "black"
                }
            }

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=map_data["LATITUDE"].mean(),
                    longitude=map_data["LONGITUDE"].mean(),
                    zoom=8,
                    pitch=0,
                ),
                layers=[layer],
                tooltip=tooltip
            ))
        else:
            st.info("Geographic coordinates not available for selected properties.")
