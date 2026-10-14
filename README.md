# CarWise — Car Price Prediction & Analytics Dashboard

## What it demonstrates
- Data cleaning and preprocessing with Pandas
- Exploratory analysis and visualizations
- Linear Regression with a train/test split
- Model evaluation with MAE and R²
- SQL tables and analytical queries using SQLite
- Interactive Streamlit UI

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`

## Important dataset note
`data/cars.csv` is a synthetic demonstration dataset generated to make the project runnable immediately. Replace it with a real used-car CSV before presenting real-world findings. Keep these columns:
- year
- selling_price
- km_driven
- fuel
- transmission

Optional columns: `car_name`, `owner`.

## Deployment
Push this folder to GitHub, then deploy `app.py` through Streamlit Community Cloud. This starter uses SQLite for simplicity. For a production-like cloud deployment with persistent records, use hosted PostgreSQL (e.g. Supabase) rather than relying on a local SQLite file.

## Project synopsis note
The provided synopsis has a title mentioning Diabetes/Car Price Prediction, while the body describes Sales Data Analysis. Confirm with your faculty which title/content is expected, then align the final report accordingly.
