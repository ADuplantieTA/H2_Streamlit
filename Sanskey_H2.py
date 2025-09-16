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
    numeric_columns = [
        "Gas Mixture Composition (X% Hydrogen Blend)",
        "Fraction of Energy from Hydrogen",
        "Energy contribution of hydrogen per 1 m³ of blended gas (X% hydrogen):",
        "Natural Gas Energy Contribution (Remaining X%)",
        "Total Energy Content of Gas Mixture",
    ]
    for col in numeric_columns:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
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

monthly_energy_data = pd.DataFrame(
    [
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
)

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

mix_column = "Gas Mixture Composition (X% Hydrogen Blend)"
fraction_column = "Fraction of Energy from Hydrogen"

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
enthalpy_h2 = abs(lookup_value("Enthalpy change of combustion H2"))
enthalpy_ch4 = abs(lookup_value("Enthalpy change of combustion CH4"))

density_ch4 = lookup_value("CH4 Gas density")
density_h2 = lookup_value("H2 Gas density")
density_o2 = lookup_value("O2 Gas density")
density_co2 = lookup_value("CO2 Gas density")
density_h2o = lookup_value("H2O Gas density")

density_map = {
    "CH₄": density_ch4,
    "H₂": density_h2,
    "O₂": density_o2,
    "CO₂": density_co2,
    "H₂O": density_h2o,
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


def render_blue_box(title: str, rows: list[tuple[str, str]], *, bg: str = "#e3f2fd", accent: str = "#1565c0") -> None:
    """Render a softly styled information box."""

    row_html = "".join(
        f'<div style=\"margin-top:8px;color:#0d1b2a;font-size:1.05rem;\"><span style=\"color:{accent};font-weight:600;\">{label}:</span> {value}</div>'
        for label, value in rows
    )
    st.markdown(
        f"""
        <div style="background-color:{bg};padding:16px;border-radius:12px;">
            <div style="color:{accent};font-weight:700;font-size:1.2rem;">{title}</div>
            {row_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def methane_combustion_summary(energy_gj: float) -> tuple[pd.DataFrame, float, float]:
    """Return mass, volume, and energy outputs for CH₄ combustion."""

    energy_gj = max(energy_gj, 0.0)
    energy_kj_abs = energy_gj * 1_000_000
    moles_ch4 = energy_kj_abs / enthalpy_ch4 if enthalpy_ch4 else 0.0

    reaction_rows = [
        ("CH₄", moles_ch4, methane_molar_mass),
        ("O₂", 2.0 * moles_ch4, oxygen_molar_mass),
        ("CO₂", moles_ch4, co2_molar_mass),
        ("H₂O", 2.0 * moles_ch4, water_molar_mass),
    ]

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


def hydrogen_combustion_summary(energy_gj: float) -> tuple[pd.DataFrame, float, float]:
    """Return mass, volume, and energy outputs for H₂ combustion."""

    energy_gj = max(energy_gj, 0.0)
    energy_kj_abs = energy_gj * 1_000_000
    moles_h2 = energy_kj_abs / enthalpy_h2 if enthalpy_h2 else 0.0

    reaction_rows = [
        ("H₂", moles_h2, hydrogen_molar_mass),
        ("O₂", 0.5 * moles_h2, oxygen_molar_mass),
        ("H₂O", moles_h2, water_molar_mass),
    ]

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

pure_ch4_table, _, pure_ch4_energy_gj = methane_combustion_summary(energy_demand_gj)
ch4_table, _, ch4_energy_gj = methane_combustion_summary(methane_energy_gj)
h2_table, _, h2_energy_gj = hydrogen_combustion_summary(hydrogen_energy_gj)

st.header("Combustion detail tables")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Methane pathway")
    st.caption(f"Energy target: {methane_energy_gj:,.2f} GJ")
    st.dataframe(ch4_table, use_container_width=True)
    st.metric(label="Energy released (GJ)", value=f"{abs(ch4_energy_gj):,.2f}")

with col2:
    st.subheader("Hydrogen pathway")
    st.caption(f"Energy target: {hydrogen_energy_gj:,.2f} GJ")
    st.dataframe(h2_table, use_container_width=True)
    st.metric(label="Energy released (GJ)", value=f"{abs(h2_energy_gj):,.2f}")

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
pure_co2_tonnes = get_component_value(pure_ch4_table, "CO₂", "Mass (tonnes)")
blend_co2_tonnes = get_component_value(ch4_table, "CO₂", "Mass (tonnes)")

water_delta_pct = percentage_change(blend_water_m3, pure_water_m3)
o2_delta_pct = percentage_change(blend_o2_m3, pure_o2_m3)
co2_delta_pct = percentage_change(blend_co2_tonnes, pure_co2_tonnes)

impact_cols = st.columns(3)

with impact_cols[0]:
    render_blue_box(
        "Water (H₂O)",
        [
            ("CH₄ baseline", f"{pure_water_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_water_m3:,.2f} m³"),
            ("Change", f"{water_delta_pct:+.1f}%"),
        ],
    )

with impact_cols[1]:
    render_blue_box(
        "Oxygen (O₂)",
        [
            ("CH₄ baseline", f"{pure_o2_m3:,.2f} m³"),
            (f"Blend {blend_percentage}%", f"{blend_o2_m3:,.2f} m³"),
            ("Change", f"{o2_delta_pct:+.1f}%"),
        ],
    )

with impact_cols[2]:
    render_blue_box(
        "Carbon dioxide (CO₂)",
        [
            ("CH₄ baseline", f"{pure_co2_tonnes:,.2f} tonnes"),
            (f"Blend {blend_percentage}%", f"{blend_co2_tonnes:,.2f} tonnes"),
            ("Change", f"{co2_delta_pct:+.1f}%"),
        ],
    )

st.subheader("Combustion comparison Sankey")

blend_h2_color = "#a6cee3"
blend_ch4_color = "#63a4ff"
oxygen_color = "#1f77b4"
combustion_color = "#c5cae9"
water_color = "#87bdd8"
co2_color = "#b0bec5"
energy_color = "#7d8b99"

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
    water_output = get_component_value(ch4_table, "H₂O", column_name) + get_component_value(
        h2_table, "H₂O", column_name
    )
    co2_output = get_component_value(ch4_table, "CO₂", column_name)
    combustion_total = water_output + co2_output

    node_labels = [
        "Hydrogen",
        "Methane",
        "Oxygen",
        "Combustion",
        "Water",
        "CO₂",
    ]

    sankey_figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".2f",
                node=dict(
                    **node_style,
                    label=node_labels,
                    color=[
                        blend_h2_color,
                        blend_ch4_color,
                        oxygen_color,
                        combustion_color,
                        water_color,
                        co2_color,
                    ],
                ),
                link=dict(
                    source=[0, 1, 2, 3, 3],
                    target=[3, 3, 3, 4, 5],
                    value=[h2_input, ch4_input, o2_input, water_output, co2_output],
                    label=[
                        format_unit_value(h2_input, unit_choice),
                        format_unit_value(ch4_input, unit_choice),
                        format_unit_value(o2_input, unit_choice),
                        format_unit_value(water_output, unit_choice),
                        format_unit_value(co2_output, unit_choice),
                    ],
                    color=[
                        color_with_alpha(blend_h2_color, 0.6),
                        color_with_alpha(blend_ch4_color, 0.6),
                        color_with_alpha(oxygen_color, 0.55),
                        color_with_alpha(water_color, 0.6),
                        color_with_alpha(co2_color, 0.6),
                    ],
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
