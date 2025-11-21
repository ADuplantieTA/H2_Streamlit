# Sanskey Flow Explorer

A Streamlit app that visualizes hydrogen/methane combustion flows with Sankey diagrams, tables, and quick condensation checks. The app reads your workbook values so the diagram stays aligned with your source data.

## Data files
- `energy_data.csv` (new): monthly service energy reference data. Columns: `Month`, `Service Energy MWh`, `Service Energy GJ`, `Input GJ @80%`.
- `Value and units.xlsx`: physical property inputs (molar masses, densities, enthalpies, etc.).
- `Mixing.xlsx`: hydrogen blending lookup table. Numeric columns are auto-cleaned for Streamlit/Arrow compatibility.

## Run locally
1. Install dependencies: `pip install streamlit pandas numpy plotly`.
2. From this folder, start the app: `streamlit run Sanskey_H2.py`.
3. Use the sidebar to set blend %, units, flue O2, and carbon intensity. Tables and Sankey links update instantly.

## Updating energy data
- Edit `energy_data.csv` to replace the default 2027 monthly values.
- Keep the same column names; the app will warn and fall back to built-in defaults if columns are missing.

## Troubleshooting
- If `Mixing.xlsx` is absent, the blend calculator is skipped (you'll see a sidebar warning).
- Ensure numeric cells in the Excel files are numbers, not strings; the loader will coerce values but empty columns are dropped.
