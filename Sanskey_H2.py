"""Streamlit Sankey diagram demo."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Sanskey Flow", layout="wide")

st.title("Sanskey Flow Explorer")
st.subheader("Author: Antoine Duplantie-Grenier")
st.caption(f"Created on {date.today():%B %d, %Y}")

st.write(
    "Use the controls to explore how combustion-related properties connect. "
    "The Sankey pulls parameter values from the accompanying Excel workbook so the diagram"
    " stays in sync with your source data."
)

DATA_FILE = Path(__file__).with_name("Value and units.xlsx")
MIXING_FILE = Path(__file__).with_name("Mixing.xlsx")
ENERGY_FILE = Path(__file__).with_name("energy_data.csv")

MIX_COMPOSITION_COL = "Gas Mixture Composition (X% Hydrogen Blend)"
MIX_FRACTION_COL = "Fraction of Energy from Hydrogen"
MIX_NUMERIC_COLUMNS = [
    MIX_COMPOSITION_COL,
    "Energy contribution of hydrogen per 1 m³ of blended gas (X% hydrogen):",
    "Natural Gas Energy Contribution (Remaining X%)",
    "Total Energy Content of Gas Mixture",
    MIX_FRACTION_COL,
    "Energy Shortfall Compared to Pure Natural Gas",
    "Additional Volume of Gas Mixture Required to Meet Original Energy Demand (Equivalent to 1 m³ NG)",
    "Natural Gas Volume Required in Gas Mixture (X% Hydrogen) to Match Energy Content of 1 m³ NG",
    "Hydrogen Volume Required in Gas Mixture (X% Hydrogen) to Match Energy Content of 1 m³ NG",
]


@st.cache_data
def load_parameter_data(path: Path) -> pd.DataFrame:
    """Return a cleaned DataFrame of parameter metadata."""

    frame = pd.read_excel(path)
    frame = frame.rename(columns=lambda col: str(col).strip())
    if "Data Parameter" in frame:
        frame["Data Parameter"] = frame["Data Parameter"].fillna("").astype(str).str.strip()
    else:
        frame["Data Parameter"] = ""
    if "Units" in frame:
        frame["Units"] = frame["Units"].fillna("").astype(str).str.strip()
    value_series = frame["Value"] if "Value" in frame else pd.Series(dtype="float64")
    frame["Value"] = pd.to_numeric(value_series, errors="coerce")
    frame = frame.dropna(how="all")
    return frame


@st.cache_data
def load_mixing_data(path: Path) -> pd.DataFrame:
    """Return hydrogen blending reference data."""

    frame = pd.read_excel(path)
    frame = frame.rename(columns=lambda col: str(col).strip())
    for col in MIX_NUMERIC_COLUMNS:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if MIX_COMPOSITION_COL in frame:
        frame = frame.dropna(subset=[MIX_COMPOSITION_COL], how="all")
    frame = frame.dropna(how="all")
    frame = frame.reset_index(drop=True)
    return frame


DEFAULT_ENERGY_ROWS = [
    {"Month": "Jan", "Service Energy MWh": 18328, "Service Energy GJ": 65982, "Input GJ @80%": 82478},
    {"Month": "Feb", "Service Energy MWh": 15063, "Service Energy GJ": 54226, "Input GJ @80%": 67782},
    {"Month": "Mar", "Service Energy MWh": 12879, "Service Energy GJ": 46364, "Input GJ @80%": 57955},
    {"Month": "Apr", "Service Energy MWh": 10984, "Service Energy GJ": 39542, "Input GJ @80%": 49427},
    {"Month": "May", "Service Energy MWh": 6147, "Service Energy GJ": 22129, "Input GJ @80%": 27662},
    {"Month": "Jun", "Service Energy MWh": 4154, "Service Energy GJ": 14954, "Input GJ @80%": 18692},
    {"Month": "Jul", "Service Energy MWh": 2549, "Service Energy GJ": 9175, "Input GJ @80%": 11469},
    {"Month": "Aug", "Service Energy MWh": 2290, "Service Energy GJ": 8244, "Input GJ @80%": 10305},
    {"Month": "Sep", "Service Energy MWh": 3345, "Service Energy GJ": 12044, "Input GJ @80%": 15055},
    {"Month": "Oct", "Service Energy MWh": 6299, "Service Energy GJ": 22675, "Input GJ @80%": 28344},
    {"Month": "Nov", "Service Energy MWh": 15820, "Service Energy GJ": 56950, "Input GJ @80%": 71188},
    {"Month": "Dec", "Service Energy MWh": 22207, "Service Energy GJ": 79945, "Input GJ @80%": 99932},
]


@st.cache_data
def load_energy_data(path: Path) -> pd.DataFrame:
    """Load monthly service energy data for the Sankey reference."""

    default_frame = pd.DataFrame(DEFAULT_ENERGY_ROWS)
    if not path.exists():
        st.warning("energy_data.csv is missing; using built-in reference values.")
        return default_frame

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive path
        st.warning(f"Could not read {path.name}; falling back to defaults. ({exc})")
        return default_frame

    frame = frame.rename(columns=lambda col: str(col).strip())
    required_cols = ["Month", "Service Energy MWh", "Service Energy GJ", "Input GJ @80%"]
    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        st.warning(f"{path.name} is missing columns: {', '.join(missing_cols)}. Using defaults instead.")
        return default_frame

    frame = frame[required_cols]
    frame["Month"] = frame["Month"].astype(str)
    for col in required_cols[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["Month"])
    frame = frame.dropna(how="all")
    return frame


if not DATA_FILE.exists():
    st.error("Value and units.xlsx is missing from the working directory.")
    st.stop()


parameter_df = load_parameter_data(DATA_FILE)

if not MIXING_FILE.exists():
    st.sidebar.warning("Mixing.xlsx is missing; blend calculator will be unavailable.")
    mixing_df = None
else:
    mixing_df = load_mixing_data(MIXING_FILE)

monthly_energy_data = load_energy_data(ENERGY_FILE)
if monthly_energy_data.empty:
    st.error("No monthly energy data available. Please populate energy_data.csv.")
    st.stop()

annual_totals = {
    "Month": "2027 Total",
    "Service Energy MWh": monthly_energy_data["Service Energy MWh"].sum(),
    "Service Energy GJ": monthly_energy_data["Service Energy GJ"].sum(),
    "Input GJ @80%": monthly_energy_data["Input GJ @80%"].sum(),
}

sidebar = st.sidebar
sidebar.header("Settings")

blend_percentage = sidebar.slider(
    "Hydrogen blend (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)

mix_column = MIX_COMPOSITION_COL
fraction_column = MIX_FRACTION_COL

hydrogen_energy_fraction = 0.0
selection_row: pd.DataFrame | None = None

if mixing_df is None or mixing_df.empty:
    sidebar.info("Add Mixing.xlsx to enable blend-driven energy split.")
else:
    blend_target = blend_percentage / 100.0
    valid_mixing = mixing_df.dropna(subset=[mix_column, fraction_column])
    if not valid_mixing.empty:
        selection_row = valid_mixing.iloc[
            (valid_mixing[mix_column] - blend_target).abs().argsort()[:1]
        ]
        value = selection_row[fraction_column].iloc[0]
        hydrogen_energy_fraction = float(value) if pd.notna(value) else 0.0

sidebar.metric(
    "Hydrogen share of energy",
    f"{hydrogen_energy_fraction * 100:.2f}%",
)

unit_choice = sidebar.selectbox(
    "Sankey unit",
    ("Volume (m3)", "Mass (tonnes)", "Energy (GJ)"),
)

# Hydrogen upstream CO2 intensity
h2_ci_kg_per_kg = sidebar.slider(
    "H₂ carbon intensity (kgCO₂e/kg H₂)",
    min_value=0.0,
    max_value=15.0,
    value=0.0,
    step=0.1,
)

flue_o2_pct = sidebar.slider(
    "Flue O₂ (vol %)",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.5,
    help="Target oxygen volume fraction in dry/wet flue gas (ideal gas basis).",
)
flue_o2_frac = flue_o2_pct / 100.0

show_o2_flue_node = sidebar.checkbox(
    "Show O₂ (flue) node in Sankey",
    value=True,
)

show_n2_flue_node = sidebar.checkbox(
    "Show N₂ (flue) node in Sankey",
    value=True,
)

dry_flue_basis = sidebar.checkbox(
    "Use dry flue basis for O₂ target",
    value=True,
    help="Exclude H₂O from flue composition when enforcing O₂ vol%.",
)




def lookup_value(keyword: str) -> float:
    """Fetch the first numeric value whose Data Parameter contains keyword."""

    mask = parameter_df["Data Parameter"].str.contains(keyword, case=False, na=False)
    values = parameter_df.loc[mask, "Value"].dropna()
    return float(values.iloc[0]) if not values.empty else 0.0


hydrogen_molar_mass = lookup_value("Di-Hydrogen molar mass")
methane_molar_mass = lookup_value("Methane molar mass")
oxygen_molar_mass = lookup_value("Dioxygen")
water_molar_mass = lookup_value("Water molar mass")
co2_molar_mass = lookup_value("CO2 molar mass")
# Nitrogen parameters (fallbacks if not present in workbook)
nitrogen_molar_mass = lookup_value("Nitrogen molar mass") or 28.0134
enthalpy_h2 = abs(lookup_value("Enthalpy change of combustion H2"))
enthalpy_ch4 = abs(lookup_value("Enthalpy change of combustion CH4"))

# Hydrogen energy density (GJ/kg, HHV)
h2_energy_density_gj_per_kg = lookup_value("H2: Energy Density")
if not h2_energy_density_gj_per_kg:
    h2_energy_density_gj_per_kg = 0.142

density_ch4 = lookup_value("CH4 Gas density")
density_h2 = lookup_value("H2 Gas density")
density_o2 = lookup_value("O2 Gas density")
density_co2 = lookup_value("CO2 Gas density")

def ideal_gas_density_kg_per_m3(molar_mass_g_per_mol: float, T_K: float = 288.15, P_Pa: float = 101325.0) -> float:
    R = 8.314462618
    M = max(1e-9, molar_mass_g_per_mol) / 1000.0
    return (P_Pa * M) / (R * max(1e-6, T_K))

density_h2o_vapor = (
    lookup_value("H2O Gas density")
    or lookup_value("Water vapor density")
    or lookup_value("Steam density")
)
if not density_h2o_vapor:
    density_h2o_vapor = ideal_gas_density_kg_per_m3(water_molar_mass, 288.15, 101325.0)

density_n2 = lookup_value("N2 Gas density") or 1.186  # kg/m³ at ~15°C, 1 atm (approx)

density_map = {
    "CH₄": density_ch4,
    "H₂": density_h2,
    "O₂": density_o2,
    "O₂ (flue)": density_o2,
    "CO₂": density_co2,
    "H₂O": density_h2o_vapor,
    "N₂": density_n2,
}


def color_with_alpha(hex_color: str, alpha: float = 0.6) -> str:
    """Return rgba string for the supplied hex color with transparency."""

    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return hex_color
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def render_blue_box(
    title: str,
    rows: list[tuple[str, str]],
    *,
    bg: str = "#e3f2fd",
    accent: str = "#1565c0",
    title_size: str = "1.2rem",
    row_size: str = "1.05rem",
    padding: str = "16px",
) -> None:
    """Render a softly styled information box with adjustable sizing."""

    row_html = "".join(
        f'<div style=\"margin-top:8px;color:#0d1b2a;font-size:{row_size};\"><span style=\"color:{accent};font-weight:600;\">{label}:</span> {value}</div>'
        for label, value in rows
    )
    st.markdown(
        f"""
        <div style="background-color:{bg};padding:{padding};border-radius:12px;">
            <div style="color:{accent};font-weight:700;font-size:{title_size};">{title}</div>
            {row_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def methane_combustion_summary(
    energy_gj: float,
    *,
    flue_o2_mole_frac: float = 0.0,
    dry_flue_basis: bool = False,
) -> tuple[pd.DataFrame, float, float]:
    """Return mass, volume, and energy outputs for CH₄ combustion.

    Includes air nitrogen and optional excess O₂ such that the flue gas has
    ``flue_o2_mole_frac`` oxygen by mole (ideal gas basis).
    """

    energy_gj = max(energy_gj, 0.0)
    energy_kj_abs = energy_gj * 1_000_000
    moles_ch4 = energy_kj_abs / enthalpy_ch4 if enthalpy_ch4 else 0.0

    # Stoichiometric consumption and production
    o2_req = 2.0 * moles_ch4
    co2_prod = moles_ch4
    h2o_prod = 2.0 * moles_ch4

    # Air composition (molar)
    air_o2_frac = 0.21
    air_n2_frac = 0.79
    n2_per_o2 = air_n2_frac / air_o2_frac  # ≈ 3.7619

    # Compute excess O₂ to reach target flue fraction (ideal gas: mole fraction == volume fraction)
    o2_excess = 0.0
    if flue_o2_mole_frac > 0.0:
        r = n2_per_o2
        # products on flue basis excluding N2 and excess O2
        P = co2_prod + (0.0 if dry_flue_basis else h2o_prod)
        denom = 1.0 - flue_o2_mole_frac * (r + 1.0)
        if denom <= 0:
            # Clamp to avoid non-physical values if user sets too high percentage
            denom = 1e-9
        o2_excess = flue_o2_mole_frac * (P + r * o2_req) / denom

    # Intake via air: O2_in includes stoich + excess; N2 passes through
    o2_in = o2_req + o2_excess
    n2_in = n2_per_o2 * o2_in

    reaction_rows = [
        ("CH₄", moles_ch4, methane_molar_mass),
        ("O₂", o2_in, oxygen_molar_mass),  # intake oxygen
        ("N₂", n2_in, nitrogen_molar_mass),  # air nitrogen
        ("CO₂", co2_prod, co2_molar_mass),
        ("H₂O", h2o_prod, water_molar_mass),
    ]
    # Represent excess O2 explicitly as a flue component
    if o2_excess > 0:
        reaction_rows.append(("O₂ (flue)", o2_excess, oxygen_molar_mass))

    molecules: list[str] = []
    moles_list: list[float] = []
    mass_g_list: list[float] = []
    mass_tonnes_list: list[float] = []
    volume_m3_list: list[float] = []

    for name, moles, molar_mass in reaction_rows:
        mass_g = moles * molar_mass
        density = density_map.get(name, 0.0)
        volume_m3 = (mass_g / 1000.0) / density if density else 0.0
        molecules.append(name)
        moles_list.append(moles)
        mass_g_list.append(mass_g)
        mass_tonnes_list.append(mass_g / 1_000_000)
        volume_m3_list.append(volume_m3)

    result = pd.DataFrame(
        {
            "Molecule": molecules,
            "Moles": moles_list,
            "Mass (g)": mass_g_list,
            "Mass (tonnes)": mass_tonnes_list,
            "Volume (m3)": volume_m3_list,
        }
    )
    energy_kj = -energy_kj_abs
    energy_gj_out = -energy_gj
    return result, energy_kj, energy_gj_out


def hydrogen_combustion_summary(
    energy_gj: float,
    *,
    flue_o2_mole_frac: float = 0.0,
    dry_flue_basis: bool = False,
) -> tuple[pd.DataFrame, float, float]:
    """Return mass, volume, and energy outputs for H₂ combustion.

    Includes air nitrogen and optional excess O₂ such that the flue gas has
    ``flue_o2_mole_frac`` oxygen by mole (ideal gas basis).
    """

    energy_gj = max(energy_gj, 0.0)
    energy_kj_abs = energy_gj * 1_000_000
    moles_h2 = energy_kj_abs / enthalpy_h2 if enthalpy_h2 else 0.0

    # Stoichiometric consumption and production
    o2_req = 0.5 * moles_h2
    h2o_prod = moles_h2

    # Air composition (molar)
    air_o2_frac = 0.21
    air_n2_frac = 0.79
    n2_per_o2 = air_n2_frac / air_o2_frac

    # Excess O2 to reach flue fraction target
    o2_excess = 0.0
    if flue_o2_mole_frac > 0.0:
        r = n2_per_o2
        # products on flue basis excluding N2 and excess O2
        P = 0.0 if dry_flue_basis else h2o_prod
        denom = 1.0 - flue_o2_mole_frac * (r + 1.0)
        if denom <= 0:
            denom = 1e-9
        o2_excess = flue_o2_mole_frac * (P + r * o2_req) / denom

    o2_in = o2_req + o2_excess
    n2_in = n2_per_o2 * o2_in

    reaction_rows = [
        ("H₂", moles_h2, hydrogen_molar_mass),
        ("O₂", o2_in, oxygen_molar_mass),
        ("N₂", n2_in, nitrogen_molar_mass),
        ("H₂O", h2o_prod, water_molar_mass),
    ]
    if o2_excess > 0:
        reaction_rows.append(("O₂ (flue)", o2_excess, oxygen_molar_mass))

    molecules: list[str] = []
    moles_list: list[float] = []
    mass_g_list: list[float] = []
    mass_tonnes_list: list[float] = []
    volume_m3_list: list[float] = []

    for name, moles, molar_mass in reaction_rows:
        mass_g = moles * molar_mass
        density = density_map.get(name, 0.0)
        volume_m3 = (mass_g / 1000.0) / density if density else 0.0
        molecules.append(name)
        moles_list.append(moles)
        mass_g_list.append(mass_g)
        mass_tonnes_list.append(mass_g / 1_000_000)
        volume_m3_list.append(volume_m3)

    result = pd.DataFrame(
        {
            "Molecule": molecules,
            "Moles": moles_list,
            "Mass (g)": mass_g_list,
            "Mass (tonnes)": mass_tonnes_list,
            "Volume (m3)": volume_m3_list,
        }
    )
    energy_kj = -energy_kj_abs
    energy_gj_out = -energy_gj
    return result, energy_kj, energy_gj_out


def get_component_value(table: pd.DataFrame, molecule: str, column: str) -> float:
    """Return a single value from the combustion summary table."""

    series = table.loc[table["Molecule"] == molecule, column]
    return float(series.iloc[0]) if not series.empty else 0.0


def percentage_change(new_value: float, baseline: float) -> float:
    """Return percentage change relative to baseline."""

    if baseline == 0:
        return 0.0
    return (new_value - baseline) / baseline * 100.0


def format_unit_value(value: float, unit: str) -> str:
    """Format flow values with a readable unit suffix."""

    suffix = {
        "Volume (m3)": "m³",
        "Mass (tonnes)": "tonnes",
        "Energy (GJ)": "GJ",
    }.get(unit, "")
    return f"{value:,.2f} {suffix}".strip()


# --- Thermodynamic helpers for condensation validation ---
def f_to_c(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0


def c_to_f(t_c: float) -> float:
    return t_c * 9.0 / 5.0 + 32.0


def sat_p_h2o_kpa(t_c: float) -> float:
    """Saturation vapor pressure of water in kPa (Buck 1981, over water)."""
    return 0.61121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def dewpoint_c_from_y(y_h2o: float, p_kpa: float) -> float:
    """Return dewpoint (°C) for given water mole fraction and total pressure (kPa)."""
    y = max(0.0, min(0.999, float(y_h2o)))
    if y <= 0.0 or p_kpa <= 0:
        return float("nan")
    target = y * p_kpa
    lo, hi = -40.0, 200.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if sat_p_h2o_kpa(mid) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def latent_heat_kj_per_kg(t_c: float) -> float:
    """Approx latent heat of vaporization at t_c (°C), kJ/kg."""
    return max(0.0, 2501.0 - 2.381 * t_c)


def condensation_result(
    n_h2o_mol: float,
    n_dry_mol: float,
    t_in_c: float,
    t_out_c: float,
    t_cond_c: float,
    p_kpa: float = 101.325,
):
    """Compute condensation and latent recovery for given flue composition and temps.

    Returns a dict with dewpoint, condensed mass, and latent/sensible recovery.
    """
    n_h2o_mol = max(0.0, float(n_h2o_mol))
    n_dry_mol = max(0.0, float(n_dry_mol))
    denom = n_h2o_mol + n_dry_mol
    y_in = (n_h2o_mol / denom) if denom > 0 else 0.0
    t_dp_c = dewpoint_c_from_y(y_in, p_kpa)

    # If outlet temp is above dewpoint, no condensation
    if not np.isfinite(t_dp_c) or t_out_c >= t_dp_c:
        return dict(
            dewpoint_c=t_dp_c,
            y_in=y_in,
            n_cond_mol=0.0,
            m_cond_kg=0.0,
            q_latent_kj=0.0,
            q_liq_sens_kj=0.0,
            q_total_kj=0.0,
        )

    # Saturation mole fraction at outlet temp
    y_sat_out = min(0.999, sat_p_h2o_kpa(t_out_c) / p_kpa)
    # Remaining vapor to be at saturation: n_w_out = y * n_dry / (1 - y)
    n_h2o_out_mol = (y_sat_out * n_dry_mol) / (1.0 - y_sat_out) if y_sat_out < 1.0 else n_h2o_mol
    n_cond_mol = max(0.0, n_h2o_mol - n_h2o_out_mol)
    m_cond_kg = n_cond_mol * (water_molar_mass / 1000.0)

    h_fg = latent_heat_kj_per_kg(t_out_c)
    cp_liq = 4.18  # kJ/kg-K
    q_latent_kj = m_cond_kg * h_fg
    q_liq_sens_kj = m_cond_kg * cp_liq * max(t_out_c - t_cond_c, 0.0)
    q_total_kj = q_latent_kj + q_liq_sens_kj

    return dict(
        dewpoint_c=t_dp_c,
        y_in=y_in,
        n_cond_mol=n_cond_mol,
        m_cond_kg=m_cond_kg,
        q_latent_kj=q_latent_kj,
        q_liq_sens_kj=q_liq_sens_kj,
        q_total_kj=q_total_kj,
    )

# Heat capacity approximations (kJ/kg-K)
CP_VAP_H2O = 1.99
CP_LIQ_H2O = 4.18
CP_N2 = 1.04
CP_O2 = 0.92
CP_CO2 = 0.85


st.subheader("2027 Service Energy Overview")

selected_row = annual_totals

monthly_cols = st.columns(3)
with monthly_cols[0]:
    render_blue_box(
        "Service energy",
        [("MWh (2027)", f"{selected_row['Service Energy MWh']:,}")],
    )
with monthly_cols[1]:
    render_blue_box(
        "Service energy",
        [("GJ (2027)", f"{selected_row['Service Energy GJ']:,}")],
    )
with monthly_cols[2]:
    render_blue_box(
        "Generation input",
        [("@ 80% eff. (GJ)", f"{selected_row['Input GJ @80%']:,}")],
    )

with st.expander("Monthly breakdown (reference)"):
    table_df = monthly_energy_data.copy()
    total_row = pd.DataFrame([annual_totals])
    st.dataframe(pd.concat([table_df, total_row], ignore_index=True), use_container_width=True)

energy_demand_gj = float(selected_row["Input GJ @80%"])
st.caption(f"2027 energy demand (80% efficiency basis): {energy_demand_gj:,.0f} GJ")

hydrogen_energy_gj = energy_demand_gj * hydrogen_energy_fraction
methane_energy_gj = energy_demand_gj - hydrogen_energy_gj

# Upstream CO2e for hydrogen based on sidebar intensity and energy density conversion
kg_co2_per_gj_h2 = (h2_ci_kg_per_kg / h2_energy_density_gj_per_kg) if h2_energy_density_gj_per_kg else 0.0
h2_upstream_co2e_tonnes = hydrogen_energy_gj * kg_co2_per_gj_h2 / 1000.0
sidebar.caption(f"Converted intensity ≈ {kg_co2_per_gj_h2:.1f} kgCO₂e/GJ H₂")

energy_col1, energy_col2 = st.columns(2)
with energy_col1:
    render_blue_box(
        "Hydrogen energy",
        [("Portion (GJ)", f"{hydrogen_energy_gj:,.2f}")],
    )
with energy_col2:
    render_blue_box(
        "Methane energy",
        [("Portion (GJ)", f"{methane_energy_gj:,.2f}")],
    )

pure_ch4_table, _, pure_ch4_energy_gj = methane_combustion_summary(
    energy_demand_gj, flue_o2_mole_frac=flue_o2_frac, dry_flue_basis=dry_flue_basis
)
ch4_table, _, ch4_energy_gj = methane_combustion_summary(
    methane_energy_gj, flue_o2_mole_frac=flue_o2_frac, dry_flue_basis=dry_flue_basis
)
h2_table, _, h2_energy_gj = hydrogen_combustion_summary(
    hydrogen_energy_gj, flue_o2_mole_frac=flue_o2_frac, dry_flue_basis=dry_flue_basis
)

st.header("Combustion detail tables")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Methane pathway")
    st.caption(f"Energy target: {methane_energy_gj:,.2f} GJ")
    st.dataframe(ch4_table, use_container_width=True)
    st.metric(label="Energy released (GJ)", value=f"{abs(ch4_energy_gj):,.2f}")
    # Stoichiometric equation with air and optional excess O2, scaled to energy target
    ch4_m_ch4 = get_component_value(ch4_table, "CH₄", "Moles")
    ch4_m_o2_in = get_component_value(ch4_table, "O₂", "Moles")
    ch4_m_n2 = get_component_value(ch4_table, "N₂", "Moles")
    ch4_m_co2 = get_component_value(ch4_table, "CO₂", "Moles")
    ch4_m_h2o = get_component_value(ch4_table, "H₂O", "Moles")
    ch4_m_o2_excess = get_component_value(ch4_table, "O₂ (flue)", "Moles")
    render_blue_box(
        "Stoichiometry",
        [
            (
                "Balanced",
                "CH₄ + 2(O₂ + 3.76 N₂) → CO₂ + 2 H₂O + 7.52 N₂",
            ),
            (
                "For target",
                (
                    f"{ch4_m_ch4:,.2f} CH₄ + {ch4_m_o2_in:,.2f} O₂ + {ch4_m_n2:,.2f} N₂ → "
                    f"{ch4_m_co2:,.2f} CO₂ + {ch4_m_h2o:,.2f} H₂O + {ch4_m_n2:,.2f} N₂"
                    + (f" + {ch4_m_o2_excess:,.2f} O₂" if ch4_m_o2_excess > 0 else "")
                ),
            ),
        ],
    )

with col2:
    st.subheader("Hydrogen pathway")
    st.caption(f"Energy target: {hydrogen_energy_gj:,.2f} GJ")
    st.dataframe(h2_table, use_container_width=True)
    st.metric(label="Energy released (GJ)", value=f"{abs(h2_energy_gj):,.2f}")
    # Stoichiometric equation with air and optional excess O2, scaled to energy target
    h2_m_h2 = get_component_value(h2_table, "H₂", "Moles")
    h2_m_o2_in = get_component_value(h2_table, "O₂", "Moles")
    h2_m_n2 = get_component_value(h2_table, "N₂", "Moles")
    h2_m_h2o = get_component_value(h2_table, "H₂O", "Moles")
    h2_m_o2_excess = get_component_value(h2_table, "O₂ (flue)", "Moles")
    render_blue_box(
        "Stoichiometry",
        [
            ("Balanced", "2 H₂ + (O₂ + 3.76 N₂) → 2 H₂O + 3.76 N₂"),
            (
                "For target",
                (
                    f"{h2_m_h2:,.2f} H₂ + {h2_m_o2_in:,.2f} O₂ + {h2_m_n2:,.2f} N₂ → "
                    f"{h2_m_h2o:,.2f} H₂O + {h2_m_n2:,.2f} N₂"
                    + (f" + {h2_m_o2_excess:,.2f} O₂" if h2_m_o2_excess > 0 else "")
                ),
            ),
        ],
    )

st.divider()

st.subheader("Blend impact summary")

pure_water_m3 = get_component_value(pure_ch4_table, "H₂O", "Volume (m3)")
blend_water_m3 = get_component_value(ch4_table, "H₂O", "Volume (m3)") + get_component_value(
    h2_table, "H₂O", "Volume (m3)"
)
pure_o2_m3 = get_component_value(pure_ch4_table, "O₂", "Volume (m3)")
blend_o2_m3 = get_component_value(ch4_table, "O₂", "Volume (m3)") + get_component_value(
    h2_table, "O₂", "Volume (m3)"
)
pure_n2_m3 = get_component_value(pure_ch4_table, "N₂", "Volume (m3)")
blend_n2_m3 = get_component_value(ch4_table, "N₂", "Volume (m3)") + get_component_value(
    h2_table, "N₂", "Volume (m3)"
)
pure_o2_flue_m3 = get_component_value(pure_ch4_table, "O₂ (flue)", "Volume (m3)")
blend_o2_flue_m3 = get_component_value(ch4_table, "O₂ (flue)", "Volume (m3)") + get_component_value(
    h2_table, "O₂ (flue)", "Volume (m3)"
)
pure_co2_tonnes = get_component_value(pure_ch4_table, "CO₂", "Mass (tonnes)")
blend_co2_tonnes = get_component_value(ch4_table, "CO₂", "Mass (tonnes)")

water_delta_pct = percentage_change(blend_water_m3, pure_water_m3)
o2_delta_pct = percentage_change(blend_o2_m3, pure_o2_m3)
o2_flue_delta_pct = percentage_change(blend_o2_flue_m3, pure_o2_flue_m3)
n2_delta_pct = percentage_change(blend_n2_m3, pure_n2_m3)
co2_delta_pct = percentage_change(blend_co2_tonnes, pure_co2_tonnes)
basis_label = "dry" if dry_flue_basis else "wet"

impact_cols = st.columns(3)

with impact_cols[0]:
    render_blue_box(
        "Steam (H₂O)",
        [
            ("CH₄ baseline", f"{pure_water_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_water_m3:,.2f} m³"),
            ("Change", f"{water_delta_pct:+.1f}%"),
        ],
        title_size="1.35rem",
        row_size="1.15rem",
        padding="20px",
    )

with impact_cols[1]:
    render_blue_box(
        "Oxygen intake (O₂)",
        [
            ("CH₄ baseline", f"{pure_o2_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_o2_m3:,.2f} m³"),
            ("Change", f"{o2_delta_pct:+.1f}%"),
        ],
        title_size="1.35rem",
        row_size="1.15rem",
        padding="20px",
    )

with impact_cols[2]:
    total_blend_co2e_tonnes = blend_co2_tonnes + h2_upstream_co2e_tonnes
    total_delta_pct = percentage_change(total_blend_co2e_tonnes, pure_co2_tonnes)
    render_blue_box(
        "Carbon dioxide (CO₂)",
        [
            ("CH₄ baseline", f"{pure_co2_tonnes:,.2f} tonnes"),
            (f"Blend CH₄", f"{blend_co2_tonnes:,.2f} tonnes"),
            ("H₂ upstream CO₂e", f"{h2_upstream_co2e_tonnes:,.2f} tonnes"),
            ("Total blend CO₂e", f"{total_blend_co2e_tonnes:,.2f} tonnes"),
            ("Total change", f"{total_delta_pct:+.1f}%"),
        ],
        title_size="1.35rem",
        row_size="1.15rem",
        padding="20px",
    )

# Second row of impact: Nitrogen intake and flue gases
impact_cols2 = st.columns(3)
with impact_cols2[0]:
    render_blue_box(
        "Nitrogen (N₂ intake)",
        [
            ("CH₄ baseline", f"{pure_n2_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_n2_m3:,.2f} m³"),
            ("Change", f"{n2_delta_pct:+.1f}%"),
            ("Flue basis", basis_label),
        ],
        title_size="1.35rem",
        row_size="1.15rem",
        padding="20px",
    )
with impact_cols2[1]:
    render_blue_box(
        f"Flue oxygen (O₂, {basis_label} basis)",
        [
            ("CH₄ baseline", f"{pure_o2_flue_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_o2_flue_m3:,.2f} m³"),
            ("Change", f"{o2_flue_delta_pct:+.1f}%"),
        ],
        title_size="1.35rem",
        row_size="1.15rem",
        padding="20px",
    )
with impact_cols2[2]:
    # Under ideal assumptions, flue N₂ equals intake N₂
    pure_n2_flue_m3 = pure_n2_m3
    blend_n2_flue_m3 = blend_n2_m3
    n2_flue_delta_pct = percentage_change(blend_n2_flue_m3, pure_n2_flue_m3)
    render_blue_box(
        f"Flue nitrogen (N₂, {basis_label} basis)",
        [
            ("CH₄ baseline", f"{pure_n2_flue_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_n2_flue_m3:,.2f} m³"),
            ("Change", f"{n2_flue_delta_pct:+.1f}%"),
        ],
        title_size="1.35rem",
        row_size="1.15rem",
        padding="20px",
    )

st.subheader("Combustion comparison Sankey")

blend_h2_color = "#a6cee3"
blend_ch4_color = "#daae0d"
oxygen_color = "#0832ee"
nitrogen_color = "#2e7d32"
o2_flue_color = "#5c7cfa"
combustion_color = "#db0b0b"
water_color = "#377492"
co2_color = "#080000"
energy_color = "#ef0000"

node_style = dict(
    pad=42,
    thickness=32,
    line=dict(color="rgba(32, 32, 32, 0.4)", width=2),
)

if unit_choice == "Energy (GJ)":
    energy_h2 = abs(h2_energy_gj)
    energy_ch4 = abs(ch4_energy_gj)
    total_energy = energy_h2 + energy_ch4

    node_labels = [
        "Hydrogen",
        "Methane",
        "Combustion",
        "Energy",
    ]

    sankey_figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".2f",
                node=dict(
                    **node_style,
                    label=node_labels,
                    color=[blend_h2_color, blend_ch4_color, combustion_color, energy_color],
                ),
                link=dict(
                    source=[0, 1, 2],
                    target=[2, 2, 3],
                    value=[energy_h2, energy_ch4, total_energy],
                    label=[
                        format_unit_value(energy_h2, unit_choice),
                        format_unit_value(energy_ch4, unit_choice),
                        format_unit_value(total_energy, unit_choice),
                    ],
                    color=[
                        color_with_alpha(blend_h2_color, 0.55),
                        color_with_alpha(blend_ch4_color, 0.55),
                        color_with_alpha(energy_color, 0.4),
                    ],
                ),
            )
        ]
    )
else:
    column_name = unit_choice
    h2_input = get_component_value(h2_table, "H₂", column_name)
    ch4_input = get_component_value(ch4_table, "CH₄", column_name)
    o2_input = get_component_value(ch4_table, "O₂", column_name) + get_component_value(
        h2_table, "O₂", column_name
    )
    n2_input = get_component_value(ch4_table, "N₂", column_name) + get_component_value(
        h2_table, "N₂", column_name
    )
    water_output = get_component_value(ch4_table, "H₂O", column_name) + get_component_value(
        h2_table, "H₂O", column_name
    )
    co2_output = get_component_value(ch4_table, "CO₂", column_name)
    o2_flue_output = get_component_value(ch4_table, "O₂ (flue)", column_name) + get_component_value(
        h2_table, "O₂ (flue)", column_name
    )
    n2_flue_output = n2_input  # nitrogen passes through

    # Build nodes dynamically and keep indices aligned
    node_labels: list[str] = []
    node_colors: list[str] = []

    idx_h2 = len(node_labels); node_labels.append("Hydrogen"); node_colors.append(blend_h2_color)
    idx_ch4 = len(node_labels); node_labels.append("Methane"); node_colors.append(blend_ch4_color)
    idx_o2 = len(node_labels); node_labels.append("Oxygen"); node_colors.append(oxygen_color)
    idx_n2 = len(node_labels); node_labels.append("Nitrogen"); node_colors.append(nitrogen_color)
    idx_comb = len(node_labels); node_labels.append("Combustion"); node_colors.append(combustion_color)
    idx_h2o = len(node_labels); node_labels.append("Steam (H₂O)"); node_colors.append(water_color)
    idx_co2 = len(node_labels); node_labels.append("CO₂ (combustion)"); node_colors.append(co2_color)

    idx_o2flue = None
    if show_o2_flue_node:
        idx_o2flue = len(node_labels); node_labels.append("O₂ (flue)"); node_colors.append(o2_flue_color)

    idx_n2flue = None
    if show_n2_flue_node:
        idx_n2flue = len(node_labels); node_labels.append("N₂ (flue)"); node_colors.append(nitrogen_color)

    idx_h2co2e = len(node_labels); node_labels.append("H₂ CO₂e (upstream)"); node_colors.append("#b71c1c")

    # Build links
    link_source = [idx_h2, idx_ch4, idx_o2, idx_n2, idx_comb, idx_comb]
    link_target = [idx_comb, idx_comb, idx_comb, idx_comb, idx_h2o, idx_co2]
    link_value = [h2_input, ch4_input, o2_input, n2_input, water_output, co2_output]
    link_label = [
        format_unit_value(h2_input, unit_choice),
        format_unit_value(ch4_input, unit_choice),
        format_unit_value(o2_input, unit_choice),
        format_unit_value(n2_input, unit_choice),
        format_unit_value(water_output, unit_choice),
        format_unit_value(co2_output, unit_choice),
    ]
    link_color = [
        color_with_alpha(blend_h2_color, 0.6),
        color_with_alpha(blend_ch4_color, 0.6),
        color_with_alpha(oxygen_color, 0.55),
        color_with_alpha(nitrogen_color, 0.55),
        color_with_alpha(water_color, 0.6),
        color_with_alpha(co2_color, 0.6),
    ]

    if idx_o2flue is not None and o2_flue_output > 0:
        link_source.append(idx_comb)
        link_target.append(idx_o2flue)
        link_value.append(o2_flue_output)
        link_label.append(format_unit_value(o2_flue_output, unit_choice))
        link_color.append(color_with_alpha(o2_flue_color, 0.55))

    if idx_n2flue is not None and n2_flue_output > 0:
        link_source.append(idx_comb)
        link_target.append(idx_n2flue)
        link_value.append(n2_flue_output)
        link_label.append(format_unit_value(n2_flue_output, unit_choice))
        link_color.append(color_with_alpha(nitrogen_color, 0.55))

    if unit_choice == "Mass (tonnes)" and h2_upstream_co2e_tonnes > 0:
        link_source.append(idx_h2)
        link_target.append(idx_h2co2e)
        link_value.append(h2_upstream_co2e_tonnes)
        link_label.append(format_unit_value(h2_upstream_co2e_tonnes, unit_choice))
        link_color.append(color_with_alpha("#b71c1c", 0.65))

    sankey_figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".2f",
                node=dict(
                    **node_style,
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=link_source,
                    target=link_target,
                    value=link_value,
                    label=link_label,
                    color=link_color,
                ),
            )
        ]
    )

sankey_figure.update_layout(
    title_text=f"Combustion comparison – {unit_choice}",
    font=dict(size=17, color="#0b1a2b"),
    margin=dict(l=40, r=40, t=70, b=30),
    hoverlabel=dict(bgcolor="white", font=dict(color="#2c3e50")),
)

st.plotly_chart(sankey_figure, use_container_width=True)

st.markdown(
    """
    ### How to use this app
    - Review the loaded Excel tables to understand the physical properties.
    - Use the sidebar to set the hydrogen blend percentage (2027 totals are used throughout).
    - Inspect the combustion tables for masses, tonnes, and volumes derived from workbook data.
    - Pick the sidebar unit selector to see how hydrogen, methane, and oxygen feed the combustion
      step and produce blended CO₂ and H₂O outputs.
    """
)

with st.expander("Reference datasets"):
    st.markdown("**Value and units.xlsx**")
    st.dataframe(parameter_df, use_container_width=True)
    if mixing_df is not None and not mixing_df.empty:
        st.markdown("**Mixing.xlsx**")
        st.dataframe(mixing_df, use_container_width=True)

with st.expander("Condensing economizer (validation)"):
    st.caption("Estimate latent heat recovery from steam condensation across the economizer.")

    colA, colB = st.columns(2)

    with colA:
        t_in_f = st.number_input("Flue gas inlet to economizer (°F)", value=412.0, step=1.0)
        t_out_f = st.number_input("Flue gas to stack after economizer (°F)", value=246.0, step=1.0)
        t_cond_f = st.number_input("Condensate outlet temperature (°F)", value=120.0, step=1.0)
        p_kpa = st.number_input("Flue gas total pressure (kPa)", value=101.325, step=0.5)
        boiler_rate_mmbtuh = st.number_input(
            "Boiler fuel input rate (MMBtu/h)", min_value=0.0, value=0.0, step=0.5,
            help="Used to convert per‑GJ latent to a rate for comparison."
        )
        reported_recovery_mmbtuh = st.number_input(
            "Reported economizer recovery (MMBtu/h)", min_value=0.0, value=2.31, step=0.1
        )
        st.markdown("**Ambient air humidity (optional)**")
        use_air_humidity = st.checkbox("Include intake air humidity", value=True)
        t_air_f = st.number_input("Intake air temperature (°F)", value=70.0, step=1.0)
        rh_air_pct = st.number_input("Intake air relative humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        p_air_kpa = st.number_input("Intake air pressure (kPa)", value=101.325, step=0.5)

    with colB:
        # Build per‑GJ flue composition for baseline and blend
        energy_gj_total = max(1e-9, abs(pure_ch4_energy_gj))
        energy_gj_blend = max(1e-9, abs(ch4_energy_gj) + abs(h2_energy_gj))

        # Baseline (pure CH4)
        n_h2o_pure_per_gj = get_component_value(pure_ch4_table, "H₂O", "Moles") / energy_gj_total
        n_co2_pure_per_gj = get_component_value(pure_ch4_table, "CO₂", "Moles") / energy_gj_total
        n_n2_pure_per_gj = get_component_value(pure_ch4_table, "N₂", "Moles") / energy_gj_total
        n_o2flue_pure_per_gj = get_component_value(pure_ch4_table, "O₂ (flue)", "Moles") / energy_gj_total
        n_dry_pure_per_gj = n_co2_pure_per_gj + n_n2_pure_per_gj + n_o2flue_pure_per_gj

        # Blend (CH4 + H2)
        n_h2o_blend_per_gj = (
            get_component_value(ch4_table, "H₂O", "Moles") + get_component_value(h2_table, "H₂O", "Moles")
        ) / energy_gj_blend
        n_co2_blend_per_gj = get_component_value(ch4_table, "CO₂", "Moles") / energy_gj_blend
        n_n2_blend_per_gj = (
            get_component_value(ch4_table, "N₂", "Moles") + get_component_value(h2_table, "N₂", "Moles")
        ) / energy_gj_blend
        n_o2flue_blend_per_gj = (
            get_component_value(ch4_table, "O₂ (flue)", "Moles") + get_component_value(h2_table, "O₂ (flue)", "Moles")
        ) / energy_gj_blend
        n_dry_blend_per_gj = n_co2_blend_per_gj + n_n2_blend_per_gj + n_o2flue_blend_per_gj

        # Intake moist air contribution (adds to water vapor only)
        def air_water_moles_per_mole_dry(t_c: float, rh_frac: float, p_tot_kpa: float) -> float:
            pws = sat_p_h2o_kpa(t_c)
            pw = max(0.0, min(rh_frac * pws, 0.999 * p_tot_kpa))
            y = pw / p_tot_kpa
            return (y / (1.0 - y)) if y < 1.0 else 0.0

        t_air_c = f_to_c(t_air_f)
        w_moles_air = air_water_moles_per_mole_dry(t_air_c, rh_air_pct / 100.0, p_air_kpa) if use_air_humidity else 0.0

        # Dry air moles per GJ (O2 intake + N2 intake)
        o2_in_pure_per_gj = get_component_value(pure_ch4_table, "O₂", "Moles") / energy_gj_total
        n2_in_pure_per_gj = get_component_value(pure_ch4_table, "N₂", "Moles") / energy_gj_total
        dry_air_pure_per_gj = o2_in_pure_per_gj + n2_in_pure_per_gj
        o2_in_blend_per_gj = (
            get_component_value(ch4_table, "O₂", "Moles") + get_component_value(h2_table, "O₂", "Moles")
        ) / energy_gj_blend
        n2_in_blend_per_gj = (
            get_component_value(ch4_table, "N₂", "Moles") + get_component_value(h2_table, "N₂", "Moles")
        ) / energy_gj_blend
        dry_air_blend_per_gj = o2_in_blend_per_gj + n2_in_blend_per_gj

        # Added H2O from humid air
        n_h2o_air_pure_per_gj = w_moles_air * dry_air_pure_per_gj
        n_h2o_air_blend_per_gj = w_moles_air * dry_air_blend_per_gj

        n_h2o_pure_per_gj_total = n_h2o_pure_per_gj + n_h2o_air_pure_per_gj
        n_h2o_blend_per_gj_total = n_h2o_blend_per_gj + n_h2o_air_blend_per_gj

        # Convert inputs
        t_in_c = f_to_c(t_in_f)
        t_out_c = f_to_c(t_out_f)
        t_cond_c = f_to_c(t_cond_f)

        # Per‑GJ basis condensation results
        res_pure = condensation_result(n_h2o_pure_per_gj_total, n_dry_pure_per_gj, t_in_c, t_out_c, t_cond_c, p_kpa)
        res_blend = condensation_result(n_h2o_blend_per_gj_total, n_dry_blend_per_gj, t_in_c, t_out_c, t_cond_c, p_kpa)

        # Vapor sensible (all water vapor entering)
        m_h2o_pure_in_kg = n_h2o_pure_per_gj_total * (water_molar_mass / 1000.0)
        m_h2o_blend_in_kg = n_h2o_blend_per_gj_total * (water_molar_mass / 1000.0)
        q_vap_sens_pure_kj = m_h2o_pure_in_kg * CP_VAP_H2O * max(t_in_c - t_out_c, 0.0)
        q_vap_sens_blend_kj = m_h2o_blend_in_kg * CP_VAP_H2O * max(t_in_c - t_out_c, 0.0)

        # Dry gas sensible
        # Mass per GJ for dry components
        m_CO2_pure_kg = (get_component_value(pure_ch4_table, "CO₂", "Mass (g)") / 1000.0) / energy_gj_total
        m_N2_pure_kg = (get_component_value(pure_ch4_table, "N₂", "Mass (g)") / 1000.0) / energy_gj_total
        m_O2f_pure_kg = (get_component_value(pure_ch4_table, "O₂ (flue)", "Mass (g)") / 1000.0) / energy_gj_total
        m_dry_pure_kg = m_CO2_pure_kg + m_N2_pure_kg + m_O2f_pure_kg
        if m_dry_pure_kg > 0:
            cp_dry_pure = (m_CO2_pure_kg * CP_CO2 + m_N2_pure_kg * CP_N2 + m_O2f_pure_kg * CP_O2) / m_dry_pure_kg
        else:
            cp_dry_pure = (CP_CO2 + CP_N2 + CP_O2) / 3.0
        q_dry_sens_pure_kj = m_dry_pure_kg * cp_dry_pure * max(t_in_c - t_out_c, 0.0)

        m_CO2_blend_kg = (get_component_value(ch4_table, "CO₂", "Mass (g)") / 1000.0) / energy_gj_blend
        m_N2_blend_kg = (
            (get_component_value(ch4_table, "N₂", "Mass (g)") + get_component_value(h2_table, "N₂", "Mass (g)")) / 1000.0
        ) / energy_gj_blend
        m_O2f_blend_kg = (
            (get_component_value(ch4_table, "O₂ (flue)", "Mass (g)") + get_component_value(h2_table, "O₂ (flue)", "Mass (g)")) / 1000.0
        ) / energy_gj_blend
        m_dry_blend_kg = m_CO2_blend_kg + m_N2_blend_kg + m_O2f_blend_kg
        if m_dry_blend_kg > 0:
            cp_dry_blend = (m_CO2_blend_kg * CP_CO2 + m_N2_blend_kg * CP_N2 + m_O2f_blend_kg * CP_O2) / m_dry_blend_kg
        else:
            cp_dry_blend = (CP_CO2 + CP_N2 + CP_O2) / 3.0
        q_dry_sens_blend_kj = m_dry_blend_kg * cp_dry_blend * max(t_in_c - t_out_c, 0.0)

        # Rate conversion if boiler firing rate provided
        GJ_per_MMBtu = 1.055056
        kJ_per_MMBtu = 1_055_056.0
        rate_gj_per_h = boiler_rate_mmbtuh * GJ_per_MMBtu
        q_water_total_pure_kj = res_pure["q_total_kj"] + q_vap_sens_pure_kj
        q_water_total_blend_kj = res_blend["q_total_kj"] + q_vap_sens_blend_kj
        q_econ_total_pure_kj = q_water_total_pure_kj + q_dry_sens_pure_kj
        q_econ_total_blend_kj = q_water_total_blend_kj + q_dry_sens_blend_kj

        q_latent_pure_mmbtuh = (res_pure["q_total_kj"] * rate_gj_per_h) / kJ_per_MMBtu if boiler_rate_mmbtuh > 0 else 0.0
        q_latent_blend_mmbtuh = (res_blend["q_total_kj"] * rate_gj_per_h) / kJ_per_MMBtu if boiler_rate_mmbtuh > 0 else 0.0
        q_water_total_pure_mmbtuh = (q_water_total_pure_kj * rate_gj_per_h) / kJ_per_MMBtu if boiler_rate_mmbtuh > 0 else 0.0
        q_water_total_blend_mmbtuh = (q_water_total_blend_kj * rate_gj_per_h) / kJ_per_MMBtu if boiler_rate_mmbtuh > 0 else 0.0
        q_econ_total_pure_mmbtuh = (q_econ_total_pure_kj * rate_gj_per_h) / kJ_per_MMBtu if boiler_rate_mmbtuh > 0 else 0.0
        q_econ_total_blend_mmbtuh = (q_econ_total_blend_kj * rate_gj_per_h) / kJ_per_MMBtu if boiler_rate_mmbtuh > 0 else 0.0

        # Display
        st.markdown("**Baseline (pure CH₄)**")
        render_blue_box(
            "Dewpoint and latent",
            [
                ("Dewpoint", f"{c_to_f(res_pure['dewpoint_c']):.1f} °F"),
                ("Outlet vs dewpoint", "condenses" if t_out_c < res_pure["dewpoint_c"] else "no condensation"),
                ("ṁ cond (kg/GJ)", f"{res_pure['m_cond_kg']:.4f}"),
                ("Q_latent+liq (kJ/GJ)", f"{res_pure['q_total_kj']:.1f}"),
                ("Q_vapor sensible (kJ/GJ)", f"{q_vap_sens_pure_kj:.1f}"),
                ("Total recovered (kJ/GJ)", f"{q_water_total_pure_kj:.1f}"),
                (
                    "Q at rate (MMBtu/h)",
                    f"latent: {q_latent_pure_mmbtuh:.2f}; water total: {q_water_total_pure_mmbtuh:.2f}",
                ),
            ],
        )

        st.markdown("**Blend (CH₄ + H₂)**")
        render_blue_box(
            "Dewpoint and latent",
            [
                ("Dewpoint", f"{c_to_f(res_blend['dewpoint_c']):.1f} °F"),
                ("Outlet vs dewpoint", "condenses" if t_out_c < res_blend["dewpoint_c"] else "no condensation"),
                ("ṁ cond (kg/GJ)", f"{res_blend['m_cond_kg']:.4f}"),
                ("Q_latent+liq (kJ/GJ)", f"{res_blend['q_total_kj']:.1f}"),
                ("Q_vapor sensible (kJ/GJ)", f"{q_vap_sens_blend_kj:.1f}"),
                ("Total recovered (kJ/GJ)", f"{q_water_total_blend_kj:.1f}"),
                (
                    "Q at rate (MMBtu/h)",
                    f"latent: {q_latent_blend_mmbtuh:.2f}; water total: {q_water_total_blend_mmbtuh:.2f}",
                ),
            ],
        )

        # Quick comparison vs reported recovery
        if boiler_rate_mmbtuh > 0:
            render_blue_box(
                "Compare to reported",
                [
                    ("Reported (MMBtu/h)", f"{reported_recovery_mmbtuh:.2f}"),
                    ("Baseline latent (MMBtu/h)", f"{q_latent_pure_mmbtuh:.2f}"),
                    ("Blend latent (MMBtu/h)", f"{q_latent_blend_mmbtuh:.2f}"),
                    ("Baseline water-only (MMBtu/h)", f"{q_water_total_pure_mmbtuh:.2f}"),
                    ("Blend water-only (MMBtu/h)", f"{q_water_total_blend_mmbtuh:.2f}"),
                    ("Baseline total incl. dry gas (MMBtu/h)", f"{q_econ_total_pure_mmbtuh:.2f}"),
                    ("Blend total incl. dry gas (MMBtu/h)", f"{q_econ_total_blend_mmbtuh:.2f}"),
                    (
                        "Note",
                        "At the provided 246°F stack temperature, latent tends to 0 (non‑condensing).",
                    ),
                ],
            )
