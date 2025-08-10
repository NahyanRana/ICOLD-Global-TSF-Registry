# Global Registry of Tailings Storage Facilities — Streamlit (pydeck)
# Professional layout; logos side-by-side; OSM basemap; neutral markers; Top-10 charts with black outlines;
# robust CSV loader; centered sub-header (+1 font size); disclaimer; taller map via CSS; no world wrap.

import io
import os
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px

st.set_page_config(page_title="Global Registry of Tailings Storage Facilities", layout="wide", page_icon="🌍")

# ====== Minimal CSS (aesthetics + taller map iframe) ======
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      h1, h2, h3 {font-weight: 700;}
      .tight {margin-top: 0.25rem; margin-bottom: 0.75rem;}

      /* Logos row */
      .logos-row {display:flex; justify-content:center; align-items:center; gap:24px; margin-bottom:10px;}
      .logos-row img {height:56px; object-fit:contain;}

      /* Make the pydeck iframe taller across Streamlit builds */
      [data-testid="stDeckGlJsonChart"] iframe,
      .stDeckGlJsonChart iframe {
      height: 900px !important;  /* was 780px */
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ====== Logos row (ICOLD left, UNEP center, CoE right; exact equal height) ======
import base64

def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logos_html = []
if os.path.exists("logo_icold.png"):
    logos_html.append(
        f"<img src='data:image/png;base64,{_img_b64('logo_icold.png')}' alt='ICOLD' style='height:56px; margin:0 12px;'/>"
    )
if os.path.exists("logo_unep.png"):
    logos_html.append(
        f"<img src='data:image/png;base64,{_img_b64('logo_unep.png')}' alt='UNEP' style='height:56px; margin:0 12px;'/>"
    )
if os.path.exists("logo_coe.png"):
    # CoE "enlarged" to match height of others — fixed to 56px for consistent alignment
    logos_html.append(
        f"<img src='data:image/png;base64,{_img_b64('logo_coe.png')}' alt='Church of England' style='height:56px; margin:0 12px;'/>"
    )

st.markdown(
    "<div style='display:flex; justify-content:center; align-items:center; gap:24px; margin-bottom:10px;'>"
    + "".join(logos_html)
    + "</div>",
    unsafe_allow_html=True,
)


# ====== Title ======
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

# ====== Sub-header (+1 font size, centered, italic) ======
st.markdown(
    """
    <div style="font-size:1.2rem; text-align:center; color:#444; font-style:italic; margin-bottom:0.5rem;">
        This initiative represents a collaboration between the International Commission on Large Dams (ICOLD) Tailings Committee, 
        the Church of England Pensions Board, and the United Nations Environment Programme. This project was launched with the goal of 
        creating a worldwide geo-census of tailings storage facilities. The registry is intended to provide factual background information 
        using consistent terminology, and to be maintained as a publicly accessible platform with up-to-date data.
    </div>
    """,
    unsafe_allow_html=True
)

# ====== Disclaimer (under sub-header) ======
st.markdown(
    """
    <div style="color:#666; font-size:0.9rem; text-align:center; margin-top:0.3rem;">
        <b>Disclaimer:</b> The data presented here is based on the most current information available at the time of compilation. 
        It is subject to ongoing verification and refinement, and may be updated or corrected as more accurate information becomes available.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Exact CSV headers we will preserve ----------
CSV_HEADERS = [
    "ID",
    "TSF Name",
    "Mine Name",
    "Current Owner",
    "Country",
    "Longitude",
    "Latitude",
    "TSF Status",
    "Number of Embankments",
    "Dam Raise Method",
    "Consequence Classification",
    "Classification System",
    "Current Max Dam Height (m)",
    "Current Storage Volume (m3)",
    "Primary Ore Commodity",
    "Year of Commencement",
    "Reference",
    "Additional Remarks",
]

# Normalize alternate names back to the exact headers above
ALT_TO_CANONICAL = {
    "ICOLD ID": "ID",
    "Name": "TSF Name",
    "Owner": "Current Owner",
    "Status": "TSF Status",
    "Height (m)": "Current Max Dam Height (m)",
    "Capacity (m3)": "Current Storage Volume (m3)",
    "Year Constructed": "Year of Commencement",
}

# Keep Dam Raise Method optional (don’t hard-require)
REQUIRED = [
    "ID",
    "TSF Name",
    "Current Owner",
    "Country",
    "Longitude",
    "Latitude",
    "TSF Status",
    "Consequence Classification",
    "Current Storage Volume (m3)",
    "Current Max Dam Height (m)",
    "Year of Commencement",
]

# ---------- Robust CSV read helpers ----------
def _clean_header(s: str) -> str:
    s = str(s)
    s = s.replace("\ufeff", "")  # BOM
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # fancy dashes -> hyphen
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

def _read_csv_any(path_or_buf):
    """Try encodings utf-8-sig / cp1252 / latin1, sniff delimiter, detect header row."""
    import io as _io
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    is_buffer = hasattr(path_or_buf, "read")
    raw_bytes = path_or_buf.read() if is_buffer else None
    last_err = None
    for enc in encodings:
        try:
            if is_buffer:
                text = raw_bytes.decode(enc, errors="replace")
                buf0 = _io.StringIO(text)
                df0 = pd.read_csv(buf0, engine="python", sep=None, header=None, skip_blank_lines=True)
                hdr = _detect_header_row(df0)
                buf = _io.StringIO(text)
                df = pd.read_csv(buf, engine="python", sep=None, header=hdr, skip_blank_lines=True)
            else:
                df0 = pd.read_csv(path_or_buf, encoding=enc, engine="python", sep=None, header=None, skip_blank_lines=True)
                hdr = _detect_header_row(df0)
                df = pd.read_csv(path_or_buf, encoding=enc, engine="python", sep=None, header=hdr, skip_blank_lines=True)
            df.columns = [_clean_header(c) for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err

@st.cache_data
def load_data(file=None) -> pd.DataFrame:
    """Load CSV, normalize column names, strip commas before numerics, keep original headers for display."""
    def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [_clean_header(c) for c in df.columns]
        rename_map = {c: ALT_TO_CANONICAL[c] for c in df.columns if c in ALT_TO_CANONICAL}
        df = df.rename(columns=rename_map)
        return df

    def _validate_and_cast(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure required columns exist
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.info(f"CSV headers detected: {list(df.columns)}")
            raise ValueError("Required columns missing.")
        # Strip thousands separators BEFORE numeric conversion
        for c in ["Longitude", "Latitude", "Current Max Dam Height (m)", "Current Storage Volume (m3)", "Year of Commencement"]:
            if c in df.columns:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop rows missing coordinates only
        df = df.dropna(subset=["Latitude", "Longitude"])
        return df

    try:
        if file is not None:
            df = _read_csv_any(file)
        else:
            df = _read_csv_any("tsf_registry_sample.csv")
        df = _normalize_headers(df)
        df = _validate_and_cast(df)
        # Reorder columns to CSV_HEADERS (keep extras at end)
        ordered = [c for c in CSV_HEADERS if c in df.columns]
        extras = [c for c in df.columns if c not in ordered]
        df = df[ordered + extras]
        return df
    except Exception as e:
        st.error(f"Could not read/validate the CSV: {e}")
        return pd.DataFrame(columns=CSV_HEADERS)

# ----- Upload -----
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded)

# Guard
if df.empty or any(c not in df.columns for c in REQUIRED):
    st.warning("No valid data loaded yet. Upload a CSV with the expected headers.")
    st.stop()

# ----- Sidebar: Filters -----
st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
owners = st.sidebar.multiselect("Current Owner", sorted(df["Current Owner"].dropna().unique()))
statuses = st.sidebar.multiselect("TSF Status", sorted(df["TSF Status"].dropna().astype(str).unique()))
classes = st.sidebar.multiselect("Consequence Classification", sorted(df["Consequence Classification"].dropna().unique()))

# Dam Raise Method filter — only if present
if "Dam Raise Method" in df.columns:
    raise_methods = st.sidebar.multiselect("Dam Raise Method", sorted(df["Dam Raise Method"].dropna().unique()))
else:
    raise_methods = []

# Marker size (constant)
marker_size = st.sidebar.slider("Marker size (m)", min_value=5_000, max_value=100_000, value=30_000, step=5_000)

# Year slider (original CSV field)
valid_years = df["Year of Commencement"].dropna().astype(int) if "Year of Commencement" in df.columns else pd.Series([], dtype=int)
year_min, year_max = (int(valid_years.min()), int(valid_years.max())) if len(valid_years) else (1900, 2030)
year_range = st.sidebar.slider("Year of Commencement", min_value=year_min, max_value=year_max, value=(year_min, year_max))

# Current Storage Volume (m3) slider
valid_vol = df["Current Storage Volume (m3)"].dropna().astype(float) if "Current Storage Volume (m3)" in df.columns else pd.Series([], dtype=float)
vol_min, vol_max = (int(valid_vol.min()), int(valid_vol.max())) if len(valid_vol) else (0, 10_000_000_000)
vol_step = max(1, (vol_max - vol_min)//50 or 1)
vol_range = st.sidebar.slider("Current Storage Volume (m3)", min_value=vol_min, max_value=vol_max, value=(vol_min, vol_max), step=vol_step)

# Name search
name_query = st.sidebar.text_input("Search TSF Name (contains):")

# ----- Apply filters (keep NaNs for Year/Volume) -----
filtered = df.copy()
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if owners:
    filtered = filtered[filtered["Current Owner"].isin(owners)]
if statuses:
    filtered = filtered[filtered["TSF Status"].astype(str).isin(statuses)]
if classes:
    filtered = filtered[filtered["Consequence Classification"].isin(classes)]
if raise_methods and "Dam Raise Method" in filtered.columns:
    filtered = filtered[filtered["Dam Raise Method"].isin(raise_methods)]
if name_query:
    filtered = filtered[filtered["TSF Name"].str.contains(name_query, case=False, na=False)]

if "Year of Commencement" in filtered.columns:
    mask_year = filtered["Year of Commencement"].between(year_range[0], year_range[1]) | filtered["Year of Commencement"].isna()
else:
    mask_year = True
if "Current Storage Volume (m3)" in filtered.columns:
    mask_vol  = filtered["Current Storage Volume (m3)"].between(vol_range[0], vol_range[1]) | filtered["Current Storage Volume (m3)"].isna()
else:
    mask_vol = True
filtered = filtered[mask_year & mask_vol]

# ----- Map (OpenStreetMap only, no wrap) -----
# Neutral styling: black fill, white outline
filtered = filtered.copy()
filtered["fill_color"] = [[0, 0, 0, 200] for _ in range(len(filtered))]
filtered["line_color"] = [[255, 255, 255] for _ in range(len(filtered))]
filtered["line_width_px"] = 1

basemap_layers = [
    pdk.Layer(
        "TileLayer",
        data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=20, tile_size=256, opacity=1.0,
        wrapLongitude=False  # prevent horizontal wrap (deck.gl option)
    )
]

# Center view
if len(filtered):
    lat0, lon0 = float(filtered["Latitude"].mean()), float(filtered["Longitude"].mean())
    zoom0 = 2.5 if len(filtered) > 3 else 4
else:
    lat0, lon0, zoom0 = 10, 0, 1.5
view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom0)

# Points layer (constant marker size), no wrap
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
    wrapLongitude=False  # prevent point wrap
)

tooltip = {"html":(
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
)}

# Use MapView with repeat disabled to avoid world wrapping
map_view = pdk.View("MapView", controller=True, repeat=False)

left_spacer, map_col, right_spacer = st.columns([1, 4, 1])

with map_col:
    st.subheader("Map")
    deck = pdk.Deck(
        layers=[*basemap_layers, layer_points],
        initial_view_state=view_state,
        views=[map_view],
        tooltip=tooltip,
        map_style=None
    )
    st.pydeck_chart(deck)


# ----- Overview Statistics -----
st.subheader("Overview Statistics")

# By Country (Top 10, black outline, no fill, thinner bars)
country_counts = (
    filtered["Country"]
    .dropna()
    .value_counts()
    .head(10)
    .sort_values(ascending=False)
    .rename_axis("Country")
    .reset_index(name="Count")
)
fig_country = px.bar(country_counts, x="Country", y="Count", title="By Country (Top 10)")
fig_country.update_traces(marker_color="rgba(0,0,0,0)", marker_line_color="black", marker_line_width=1.5)
fig_country.update_layout(showlegend=False, bargap=0.35)
st.plotly_chart(fig_country, use_container_width=True)

# By TSF Status (Top 10, black outline, no fill, thinner bars)
status_counts = (
    filtered["TSF Status"]
    .fillna("Unknown")
    .astype(str)
    .value_counts()
    .head(10)
    .sort_values(ascending=False)
    .rename_axis("TSF Status")
    .reset_index(name="Count")
)
fig_status = px.bar(status_counts, x="TSF Status", y="Count", title="By TSF Status (Top 10)")
fig_status.update_traces(marker_color="rgba(0,0,0,0)", marker_line_color="black", marker_line_width=1.5)
fig_status.update_layout(showlegend=False, bargap=0.35)
st.plotly_chart(fig_status, use_container_width=True)

# By Dam Raise Method (Top 10, black outline) — only if column exists
if "Dam Raise Method" in filtered.columns:
    raise_counts = (
        filtered["Dam Raise Method"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .sort_values(ascending=False)
        .rename_axis("Dam Raise Method")
        .reset_index(name="Count")
    )
    fig_raise = px.bar(raise_counts, x="Dam Raise Method", y="Count", title="By Dam Raise Method (Top 10)")
    fig_raise.update_traces(marker_color="rgba(0,0,0,0)", marker_line_color="black", marker_line_width=1.5)
    fig_raise.update_layout(showlegend=False, bargap=0.35)
    st.plotly_chart(fig_raise, use_container_width=True)

# ----- Data Summary Table + download (original CSV headers; hide styling columns) -----
st.subheader("Complete Database (Draft, v. June 2025)")
hide_cols = ["fill_color", "line_color", "line_width_px"]
display_cols = [c for c in CSV_HEADERS if c in filtered.columns] + [c for c in filtered.columns if c not in CSV_HEADERS + hide_cols]
display_df = filtered[display_cols]

# Pretty-print Current Storage Volume (m3) if present
if "Current Storage Volume (m3)" in display_df.columns:
    fmt_df = display_df.copy()
    fmt_df["Current Storage Volume (m3)"] = fmt_df["Current Storage Volume (m3)"].map(
        lambda x: f"{int(x):,}" if pd.notnull(x) else ""
    )
    st.dataframe(fmt_df)
else:
    st.dataframe(display_df)

st.download_button(
    "Download filtered CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="tsf_registry_filtered.csv",
    mime="text/csv"
)
