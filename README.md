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

## Hydrogen CO2e calculation (explained)
1) User sets H2 carbon intensity: `CI_H2` [kgCO2e/kg H2].  
2) Energy density of H2: `ED_H2` [GJ/kg H2] (pulled from the workbook, default 0.142 GJ/kg).  
3) Convert to per-GJ intensity: `CI_GJ = CI_H2 / ED_H2` [kgCO2e/GJ H2].  
4) Hydrogen energy share: `E_H2 = blend_fraction * total_energy` [GJ].  
5) Upstream CO2e from H2: `CO2e_H2 = (CI_GJ * E_H2) / 1000` [tonnes].  
Example with defaults: `CI_H2 = 7.715 kg/kg`, `ED_H2 = 0.142 GJ/kg` → `CI_GJ ≈ 54.33 kgCO2e/GJ`.

## Updating energy data
- Edit `energy_data.csv` to replace the default 2027 monthly values.
- Keep the same column names; the app will warn and fall back to built-in defaults if columns are missing.

## Troubleshooting
- If `Mixing.xlsx` is absent, the blend calculator is skipped (you'll see a sidebar warning).
- Ensure numeric cells in the Excel files are numbers, not strings; the loader will coerce values but empty columns are dropped.
