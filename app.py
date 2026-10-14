import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "cars.csv"
DB_PATH = BASE_DIR / "carwise.db"

st.set_page_config(page_title="CarWise | Price Analytics", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = [c.strip().lower() for c in data.columns]
    required = ["year", "selling_price", "km_driven", "fuel", "transmission"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data.drop_duplicates().copy()
    for col in ["year", "selling_price", "km_driven"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=required)
    data = data[(data["selling_price"] > 0) & (data["km_driven"] >= 0)]
    data["fuel"] = data["fuel"].astype(str).str.strip().str.title()
    data["transmission"] = data["transmission"].astype(str).str.strip().str.title()
    if "car_name" not in data.columns:
        data["car_name"] = "Car"
    if "owner" not in data.columns:
        data["owner"] = "Unknown"
    data["car_age"] = 2026 - data["year"]
    return data

def init_db(data):
    with sqlite3.connect(DB_PATH) as conn:
        data.to_sql("cars", conn, if_exists="replace", index=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                km_driven INTEGER NOT NULL,
                fuel TEXT NOT NULL,
                transmission TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def train_model(data):
    features = ["year", "km_driven", "fuel", "transmission"]
    X = data[features]
    y = data["selling_price"]
    categorical = ["fuel", "transmission"]
    numeric = ["year", "km_driven"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, mean_absolute_error(y_test, predictions), r2_score(y_test, predictions)

try:
    data = load_data()
    init_db(data)
    model, mae, r2 = train_model(data)
except Exception as e:
    st.error(f"Could not load dataset or initialise project: {e}")
    st.stop()

st.title("🚗 CarWise")
st.caption("Used-car price prediction and data analytics dashboard")


with st.sidebar:
    st.header("Filters")
    fuels = ["All"] + sorted(data["fuel"].dropna().unique().tolist())
    selected_fuel = st.selectbox("Fuel type", fuels)
    min_year = int(data["year"].min())
    max_year = int(data["year"].max())
    selected_years = st.slider("Manufacturing year", min_year, max_year, (min_year, max_year))
    st.divider()
    

filtered = data.copy()
if selected_fuel != "All":
    filtered = filtered[filtered["fuel"] == selected_fuel]
filtered = filtered[filtered["year"].between(selected_years[0], selected_years[1])]

tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🤖 Predict Price", "🗄️ SQL & Prediction History"])

with tab1:
    st.subheader("Business overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cars in filtered data", f"{len(filtered):,}")
    c2.metric("Average selling price", f"₹{filtered['selling_price'].mean():,.0f}" if len(filtered) else "—")
    c3.metric("Median selling price", f"₹{filtered['selling_price'].median():,.0f}" if len(filtered) else "—")
    c4.metric("Average distance driven", f"{filtered['km_driven'].mean():,.0f} km" if len(filtered) else "—")

    left, right = st.columns(2)
    with left:
        if len(filtered):
            by_fuel = filtered.groupby("fuel", as_index=False)["selling_price"].mean().sort_values("selling_price", ascending=False)
            fig = px.bar(by_fuel, x="fuel", y="selling_price", title="Average price by fuel type",
                         labels={"fuel":"Fuel type","selling_price":"Average price (₹)"}, text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if len(filtered):
            by_year = filtered.groupby("year", as_index=False)["selling_price"].mean()
            fig = px.line(by_year, x="year", y="selling_price", markers=True, title="Average price by manufacturing year",
                          labels={"year":"Manufacturing year","selling_price":"Average price (₹)"})
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if len(filtered):
            fig = px.scatter(filtered, x="km_driven", y="selling_price", color="fuel",
                             title="Mileage vs selling price",
                             labels={"km_driven":"Kilometers driven","selling_price":"Selling price (₹)"})
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if len(filtered):
            counts = filtered["transmission"].value_counts().rename_axis("transmission").reset_index(name="car_count")
            fig = px.pie(counts, names="transmission", values="car_count", title="Cars by transmission")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample records")
    st.dataframe(filtered.head(20), use_container_width=True)
    with st.expander("Analytics insights"):
        if len(filtered):
            top_fuel = filtered.groupby("fuel")["selling_price"].mean().idxmax()
            st.write(f"- **Highest average price by fuel type:** {top_fuel}.")
            corr = filtered["km_driven"].corr(filtered["selling_price"]) if len(filtered) > 1 else np.nan
            st.write(f"- **Correlation between kilometers driven and price:** {corr:.2f}" if pd.notna(corr) else "- Not enough data to calculate correlation.")
            st.write("- Correlation describes association in this dataset; it does not prove that mileage alone causes price changes.")

with tab2:
    st.subheader("Estimate a car's resale price")
    st.write("Enter the car details below. This is an educational estimate, not an official valuation.")
    a, b = st.columns(2)
    with a:
        year = st.number_input("Manufacturing year", min_value=1990, max_value=2026, value=2020, step=1)
        km_driven = st.number_input("Kilometers driven", min_value=0, max_value=1000000, value=45000, step=1000)
    with b:
        fuel = st.selectbox("Fuel type", sorted(data["fuel"].unique().tolist()), key="predict_fuel")
        transmission = st.selectbox("Transmission", sorted(data["transmission"].unique().tolist()), key="predict_trans")
    if st.button("Predict car price", type="primary"):
        sample = pd.DataFrame([{"year": year, "km_driven": km_driven, "fuel": fuel, "transmission": transmission}])
        predicted = max(0, float(model.predict(sample)[0]))
        st.success(f"Estimated resale price: ₹{predicted:,.0f}")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO predictions (year, km_driven, fuel, transmission, predicted_price)
                VALUES (?, ?, ?, ?, ?)
            """, (int(year), int(km_driven), fuel, transmission, predicted))
            conn.commit()
        st.caption(f"Model evaluation on held-out test data: MAE ₹{mae:,.0f} | R² {r2:.2f}")

    st.subheader("Model evaluation")
    m1, m2 = st.columns(2)
    m1.metric("Mean Absolute Error (MAE)", f"₹{mae:,.0f}")
    m2.metric("R² score", f"{r2:.2f}")
    st.caption("Metrics are measured on a held-out test split. Their reliability depends on dataset quality and representativeness.")

with tab3:
    st.subheader("SQL database")
    st.write("The app loads the cleaned car records into the SQLite `cars` table. Every prediction is stored in `predictions`.")
    query_options = {
        "Average price by fuel": """
            SELECT fuel, ROUND(AVG(selling_price), 0) AS average_price, COUNT(*) AS car_count
            FROM cars GROUP BY fuel ORDER BY average_price DESC
        """,
        "Count by transmission": """
            SELECT transmission, COUNT(*) AS total_cars
            FROM cars GROUP BY transmission ORDER BY total_cars DESC
        """,
        "Most recent predictions": """
            SELECT id, year, km_driven, fuel, transmission, ROUND(predicted_price, 0) AS predicted_price, created_at
            FROM predictions ORDER BY created_at DESC LIMIT 20
        """
    }
    chosen_query = st.selectbox("Choose an SQL analysis", list(query_options.keys()))
    with sqlite3.connect(DB_PATH) as conn:
        sql_result = pd.read_sql_query(query_options[chosen_query], conn)
    st.dataframe(sql_result, use_container_width=True)
    with sqlite3.connect(DB_PATH) as conn:
        total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    st.metric("Predictions saved in SQL", total_predictions)

st.divider()

