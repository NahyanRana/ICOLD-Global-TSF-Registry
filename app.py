# app.py
# Global Registry of Tailings Storage Facilities — Streamlit (pydeck)
# Public-only, local CSV loader; theme toggle (Light/Dark) from sidebar;
# Dark theme: white text + white chart outlines + logos on white background; map unchanged (OSM).

import io
import os
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Global Registry of Tailings Storage Facilities", layout="wide", page_icon="🌍")

# -------------------------
# FORCE PUBLIC MODE (no uploader)
# -------------------------
DEV_MODE = False  # public-only; do not show uploader or data-source debug

# Paths
APP_DIR = Path(__file__).parent.resolve()
DATA_PATH = APP_DIR / "data" / "v2.0-Global-TSF-Registry_June2025.csv"

# -------------------------
# SIDEBAR THEME (works immediately)
# -------------------------
st.sidebar.header("Display")
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, help="Switch text & charts; map remains unchanged")

# Variables derived from theme
is_dark = (theme_choice == "Dark")
plotly_template = "plotly_dark" if is_dark else "plotly"
bar_outline = "white" if is_dark else "black"
text_color = "#FFFFFF" if is_dark else "#222222"
subtext_color = "#DDDDDD" if is_dark else "#666666"

# -------------------------
# THEME-SPECIFIC CSS (text colors, logo background in dark)
# -------------------------
base_css = f"""
<style>
  .block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
  h1, h2, h3 {{font-weight: 700; color: {text_color};}}
  .tight {{margin-top: 0.25rem; margin-bottom: 0.75rem;}}
  .body-text, p, .stMarkdown, .stText, .stCaption, label, span, div {{
    color: {text_color} !important;
  }}
  .muted {{ color: {subtext_color} !important; }}
  /* Logos row (baseline) */
  .logos-row {{display:flex; justify-content:center; align-items:center; gap:28px; margin-bottom:12px;}}
  .logos-row img {{height: 76px; object-fit: contain;}}
  /* Map height */
  [data-testid="stDeckGlJsonChart"] iframe,
  .stDeckGlJsonChart iframe {{height: 900px !important;}}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# Add a white background “pill” behind logos only in Dark mode
if is_dark:
    logos_wrapper_style = "background:#FFFFFF; padding:12px 16px; border-radius:12px;"
else:
    logos_wrapper_style = ""  # transparent in Light mode

# -------------------------
# LOGOS (bigger + top spacing, white bg pill in dark)
# -------------------------
def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# top spacer so logos never get clipped
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

logos_html_items = []
if (APP_DIR / "logo_icold.png").exists():
    logos_html_items.append(f"<img src='data:image/png;base64,{_img_b64(APP_DIR / 'logo_icold.png')}' alt='ICOLD'/>")
if (APP_DIR / "logo_unep.png").exists():
    logos_html_items.append(f"<img src='data:image/png;base64,{_img_b64(APP_DIR / 'logo_unep.png')}' alt='UNEP'/>")
if (APP_DIR / "logo_coe.png").exists():
    logos_html_items.append(f"<img src='data:image/png;base64,{_img_b64(APP_DIR / 'logo_coe.png')}' alt='Church of England'/>")

if logos_html_items:
    st.markdown(
        f"<div class='logos-row' style='{logos_wrapper_style}'>" + "".join(logos_html_items) + "</div>",
        unsafe_allow_html=True
    )

# -------------------------
# TITLE + DRAFT LABEL + SUBHEADER + DISCLAIMER
# -------------------------
st.markdown(
    """
    <div style="text-align:center;">
      <h1 class="tight"><b>Global Registry of Tailings Storage Facilities</b></h1>
      <div style="font-size:1.6rem; font-weight:700; color:gray; font-style:italic;">
        Draft (v. June 2025)
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:1.1rem; text-align:center; font-style:italic; margin-bottom:0.6rem;">
        This initiative represents a collaboration between the International Commission on Large Dams (ICOLD) Tailings Committee, 
        the Church of England Pensions Board, and the United Nations Environment Programme. This project was launched with the goal of 
        creating a worldwide geo-census of tailings storage facilities. The registry is intended to provide factual background information 
        using consistent terminology, and to be maintained as a publicly accessible platform with up-to-date data.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="muted" style="font-size:0.95rem; text-align:center; margin-top:0.3rem;">
        <b>Disclaimer:</b> The data presented here is based on the most current information available at the time of compilation. 
        It is subject to ongoing verification and refinement, and may be updated or corrected as more accurate information becomes available.
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# EXPECTED HEADERS + LOADER
# -------------------------
CSV_HEADERS = [
    "ID", "TSF Name", "Mine Name", "Current Owner", "Country",
    "Longitude", "Latitude", "TSF Status", "Number of Embankments",
    "Dam Raise Method", "Consequence Classification", "Classification System",
    "Current Max Dam Height (m)", "Current Storage Volume (m3)",
    "Primary Ore Commodity", "Year of Commencement", "Reference", "Additional Remarks",
]

ALT_TO_CANONICAL = {
    "ICOLD ID": "ID",
    "Name": "TSF Name",
    "Owner": "Current Owner",
    "Status": "TSF Status",
    "Height (m)": "Current Max Dam Height (m)",
    "Capacity (m3)": "Current Storage Volume (m3)",
    "Year Constructed": "Year of Commencement",
}

REQUIRED = [
    "ID", "TSF Name", "Current Owner", "Country",
    "Longitude", "Latitude", "TSF Status", "Consequence Classification",
    "Current Storage Volume (m3)", "Current Max Dam Height (m)", "Year of Commencement",
]

def _clean_header(s: str) -> str:
    s = str(s)
    s = s.replace("\ufeff", "").replace("\u2013", "-").replace("\u2014", "-")
    s = " ".join(s.split())
    return s.strip()

def _detect_header_row(df0: pd.DataFrame) -> int:
    expected = set(_clean_header(h) for h in CSV_HEADERS)
    max_rows = min(30, len(df0))
    best_i, best_score = 0, -1
    for i in range(max_rows):
        row = [_clean_header(x) for x in df0.iloc[i].tolist()]
        score = sum(1 for x in row if x in expected)
        if score > best_score:
            best_score, best_i = score, i
    return best_i if best_score >= 5 else 0

def _read_csv_any_local(path: Path) -> pd.DataFrame:
    """Robust local CSV reader: try encodings and delimiter auto-detect; no URL."""
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df0 = pd.read_csv(path, encoding=enc, engine="python", sep=None, header=None, skip_blank_lines=True)
            hdr = _detect_header_row(df0)
            df = pd.read_csv(path, encoding=enc, engine="python", sep=None, header=hdr, skip_blank_lines=True)
            df.columns = [_clean_header(c) for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err

@st.cache_data
def load_data_local(path: Path) -> pd.DataFrame:
    df = _read_csv_any_local(path)
    # Normalize headers
    df.columns = [_clean_header(c) for c in df.columns]
    rename_map = {c: ALT_TO_CANONICAL[c] for c in df.columns if c in ALT_TO_CANONICAL}
    df = df.rename(columns=rename_map)

    # Validate required columns
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(f"CSV headers detected: {list(df.columns)}")
        raise ValueError("Required columns missing.")

    # Strip separators, cast numerics
    for c in ["Longitude", "Latitude", "Current Max Dam Height (m)", "Current Storage Volume (m3)", "Year of Commencement"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Reorder to CSV_HEADERS (keep extras at end)
    ordered = [c for c in CSV_HEADERS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]

# -------------------------
# DATA SOURCE (LOCAL FILE ONLY, no sidebar debug in public)
# -------------------------
if not DATA_PATH.exists():
    st.error(f"Registry CSV not found at: {DATA_PATH}")
    try:
        files = [p.name for p in (APP_DIR / 'data').glob('*')]
        st.error(f"Files in /data: {files}")
    except Exception as e:
        st.error(f"Could not list /data. Error: {e}")
    st.stop()

df = load_data_local(DATA_PATH)

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")

countries = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
owners = st.sidebar.multiselect("Current Owner", sorted(df["Current Owner"].dropna().unique()))
statuses = st.sidebar.multiselect("TSF Status", sorted(df["TSF Status"].dropna().astype(str).unique()))
classes = st.sidebar.multiselect("Consequence Classification", sorted(df["Consequence Classification"].dropna().unique()))
raise_methods = st.sidebar.multiselect("Dam Raise Method", sorted(df["Dam Raise Method"].dropna().unique())) if "Dam Raise Method" in df.columns else []

# Marker size constant (meters)
marker_size = st.sidebar.slider("Marker size (m)", min_value=5_000, max_value=100_000, value=30_000, step=5_000)

# Sliders
valid_years = df["Year of Commencement"].dropna().astype(int) if "Year of Commencement" in df.columns else pd.Series([], dtype=int)
year_min, year_max = (int(valid_years.min()), int(valid_years.max())) if len(valid_years) else (1900, 2030)
year_range = st.sidebar.slider("Year of Commencement", min_value=year_min, max_value=year_max, value=(year_min, year_max))

valid_vol = df["Current Storage Volume (m3)"].dropna().astype(float) if "Current Storage Volume (m3)" in df.columns else pd.Series([], dtype=float)
vol_min, vol_max = (int(valid_vol.min()), int(valid_vol.max())) if len(valid_vol) else (0, 10_000_000_000)
vol_step = max(1, (vol_max - vol_min)//50 or 1)
vol_range = st.sidebar.slider("Current Storage Volume (m3)", min_value=vol_min, max_value=vol_max, value=(vol_min, vol_max), step=vol_step)

name_query = st.sidebar.text_input("Search TSF Name (contains):")

# Apply filters
filtered = df.copy()
if countries:     filtered = filtered[filtered["Country"].isin(countries)]
if owners:        filtered = filtered[filtered["Current Owner"].isin(owners)]
if statuses:      filtered = filtered[filtered["TSF Status"].astype(str).isin(statuses)]
if classes:       filtered = filtered[filtered["Consequence Classification"].isin(classes)]
if raise_methods and "Dam Raise Method" in filtered.columns:
    filtered = filtered[filtered["Dam Raise Method"].isin(raise_methods)]
if name_query:
    filtered = filtered[filtered["TSF Name"].str.contains(name_query, case=False, na=False)]

mask_year = filtered["Year of Commencement"].between(year_range[0], year_range[1]) | filtered["Year of Commencement"].isna() if "Year of Commencement" in filtered.columns else True
mask_vol  = filtered["Current Storage Volume (m3)"].between(vol_range[0], vol_range[1]) | filtered["Current Storage Volume (m3)"].isna() if "Current Storage Volume (m3)" in filtered.columns else True
filtered = filtered[mask_year & mask_vol]

# -------------------------
# MAP (OSM tiles, no wrap, centered column) — map stays the same in both themes
# -------------------------
# Neutral styling: black fill, white outline (outline stays white even in dark to pop on OSM)
filtered = filtered.copy()
filtered["fill_color"] = [[0, 0, 0, 200] for _ in range(len(filtered))]
filtered["line_color"] = [[255, 255, 255] for _ in range(len(filtered))]
filtered["line_width_px"] = 1

# Map tiles: keep constant regardless of theme (per your request)
tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

basemap_layers = [
    pdk.Layer(
        "TileLayer",
        data=tile_url,
        min_zoom=0, max_zoom=20, tile_size=256, opacity=1.0,
        wrapLongitude=False  # prevent horizontal wrap
    )
]

if len(filtered):
    lat0, lon0 = float(filtered["Latitude"].mean()), float(filtered["Longitude"].mean())
    zoom0 = 2.5 if len(filtered) > 3 else 4
else:
    lat0, lon0, zoom0 = 10, 0, 1.5
view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom0)
map_view = pdk.View("MapView", controller=True, repeat=False)

layer_points = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position='[Longitude, Latitude]',
    get_fill_color="fill_color",
    get_radius=marker_size,
    stroked=True,
    get_line_color="line_color",
    get_line_width="line_width_px",
    pickable=True,
    wrapLongitude=False
)

left_spacer, map_col, right_spacer = st.columns([1, 5, 1])
with map_col:
    st.subheader("Map")
    deck = pdk.Deck(
        layers=[*basemap_layers, layer_points],
        initial_view_state=view_state,
        views=[map_view],
        tooltip={"html":(
            "<b>{TSF Name}</b><br/>"
            "ID: {ID}<br/>"
            "Mine: {Mine Name}<br/>"
            "Owner: {Current Owner}<br/>"
            "Country: {Country}<br/>"
            "Consequence: {Consequence Classification}<br/>"
            "Status: {TSF Status}<br/>"
            "Dam Raise: {Dam Raise Method}<br/>"
            "Year of Commencement: {Year of Commencement}<br/>"
            "Current Storage Volume (m³): {Current Storage Volume (m3)}<br/>"
            "Current Max Dam Height (m): {Current Max Dam Height (m)}"
        )},
        map_style=None  # use our TileLayer instead of Mapbox styles
    )
    st.pydeck_chart(deck)
    st.caption("<span class='muted'>Basemap © OpenStreetMap contributors</span>", unsafe_allow_html=True)

# -------------------------
# OVERVIEW STATISTICS (Top 10; outlines switch with theme)
# -------------------------
st.subheader("Overview Statistics")

country_counts = (
    filtered["Country"].dropna().value_counts().head(10).sort_values(ascending=False).rename_axis("Country").reset_index(name="Count")
)
fig_country = px.bar(country_counts, x="Country", y="Count", title="By Country (Top 10)", template=plotly_template)
fig_country.update_traces(marker_color="rgba(0,0,0,0)", marker_line_color=bar_outline, marker_line_width=1.5)
fig_country.update_layout(showlegend=False, bargap=0.35)
st.plotly_chart(fig_country, use_container_width=True)

status_counts = (
    filtered["TSF Status"].fillna("Unknown").astype(str).value_counts().head(10).sort_values(ascending=False).rename_axis("TSF Status").reset_index(name="Count")
)
fig_status = px.bar(status_counts, x="TSF Status", y="Count", title="By TSF Status (Top 10)", template=plotly_template)
fig_status.update_traces(marker_color="rgba(0,0,0,0)", marker_line_color=bar_outline, marker_line_width=1.5)
fig_status.update_layout(showlegend=False, bargap=0.35)
st.plotly_chart(fig_status, use_container_width=True)

if "Dam Raise Method" in filtered.columns:
    raise_counts = (
        filtered["Dam Raise Method"].dropna().astype(str).value_counts().head(10).sort_values(ascending=False).rename_axis("Dam Raise Method").reset_index(name="Count")
    )
    fig_raise = px.bar(raise_counts, x="Dam Raise Method", y="Count", title="By Dam Raise Method (Top 10)", template=plotly_template)
    fig_raise.update_traces(marker_color="rgba(0,0,0,0)", marker_line_color=bar_outline, marker_line_width=1.5)
    fig_raise.update_layout(showlegend=False, bargap=0.35)
    st.plotly_chart(fig_raise, use_container_width=True)

# -------------------------
# COMPLETE DATABASE TABLE + DOWNLOAD
# -------------------------
st.subheader("Database (Draft, v. June 2025)")
hide_cols = ["fill_color", "line_color", "line_width_px"]
display_cols = [c for c in CSV_HEADERS if c in filtered.columns] + [c for c in filtered.columns if c not in CSV_HEADERS + hide_cols]
display_df = filtered[display_cols]

if "Current Storage Volume (m3)" in display_df.columns:
    fmt_df = display_df.copy()
    fmt_df["Current Storage Volume (m3)"] = fmt_df["Current Storage Volume (m3)"].map(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
    st.dataframe(fmt_df)
else:
    st.dataframe(display_df)

st.download_button(
    "Download filtered CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="tsf_registry_filtered.csv",
    mime="text/csv"
)
