# app.py

import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Streamlit basic config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Ontario Permit-to-Take-Water Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# DATA LOADING & BASIC CLEANING
# ---------------------------------------------------------------------

BASE_PARQUET = "data/processed/pttw_analysis_ready.parquet"
BASE_CSV = "data/processed/pttw_analysis_ready.csv"
RAW_PATH = "data/raw/PermitsToTakeWater.csv"

RAW_DATE_COLS = ["IssuedDate", "ExpiryDate", "RenewDate", "Permit_End"]


@st.cache_data(show_spinner=True)
def load_base_data() -> pd.DataFrame:
    """
    Load the main analysis-ready dataset.

    Priority:
    1. Use processed parquet/CSV from data/processed if available.
    2. Otherwise, read data/raw/PermitsToTakeWater.csv and do core cleaning.
    """
    # 1) processed
    if os.path.exists(BASE_PARQUET):
        df = pd.read_parquet(BASE_PARQUET)
    elif os.path.exists(BASE_CSV):
        df = pd.read_csv(
            BASE_CSV,
            parse_dates=["issued_date", "expiry_date", "renew_date", "permit_end_date"],
        )
    else:
        # 2) raw fallback
        if not os.path.exists(RAW_PATH):
            st.error(
                "❌ Could not find any processed dataset in `data/processed/`, "
                "and the raw file `data/raw/PermitsToTakeWater.csv` is also missing.\n\n"
                "To fix this, add one of these:\n"
                "• `data/processed/pttw_analysis_ready.parquet` (preferred), or\n"
                "• `data/raw/PermitsToTakeWater.csv`."
            )
            st.stop()

        df = pd.read_csv(RAW_PATH, encoding="latin1")

        # parse original date columns
        for col in RAW_DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # standardise column names
        df.columns = (
            df.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("/", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.lower()
        )

        # rename to analysis-friendly names
        rename_map = {
            "maxl_day": "max_l_per_day",
            "days_year": "days_per_year",
            "hrs_daymax": "max_hours_per_day",
            "l_minute": "l_per_minute",
            "issueddate": "issued_date",
            "expirydate": "expiry_date",
            "renewdate": "renew_date",
            "permit_end": "permit_end_date",
            "surfgrnd": "surface_or_ground",
            "purposecat": "purpose_category",
            "spurpose": "specific_purpose",
        }
        df = df.rename(columns=rename_map)

        # numeric cleaning
        num_cols = ["max_l_per_day", "days_per_year", "max_hours_per_day", "l_per_minute"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "max_l_per_day" in df.columns:
            df.loc[df["max_l_per_day"] <= 0, "max_l_per_day"] = np.nan

        # engineered features
        if {"max_l_per_day", "days_per_year"}.issubset(df.columns):
            df["max_l_per_year"] = df["max_l_per_day"] * df["days_per_year"]
            df["max_m3_per_day"] = df["max_l_per_day"] / 1000
            df["max_m3_per_year"] = df["max_l_per_year"] / 1000

        if {"issued_date", "expiry_date"}.issubset(df.columns):
            df["permit_duration_days"] = (df["expiry_date"] - df["issued_date"]).dt.days
            df["permit_duration_years"] = df["permit_duration_days"] / 365.25

        if "surface_or_ground" in df.columns:
            upper = df["surface_or_ground"].astype("string").str.upper()
            df["is_surface_water"] = np.where(upper.str.startswith("S"), 1, 0)
            df["is_groundwater"] = np.where(upper.str.startswith("G"), 1, 0)

        if "active" in df.columns:
            df["active"] = df["active"].astype("string").str.title()

        df = df.drop_duplicates()

    # ensure datetime + handy time columns
    for col in ["issued_date", "expiry_date", "renew_date", "permit_end_date"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "issued_date" in df.columns and "issued_year" not in df.columns:
        df["issued_year"] = df["issued_date"].dt.year
    if "issued_date" in df.columns and "issued_year_month" not in df.columns:
        df["issued_year_month"] = df["issued_date"].dt.to_period("M").astype(str)

    return df


pttw = load_base_data()


# ---------------------------------------------------------------------
# SUMMARY BUILDERS (ALL FROM pttw)
# ---------------------------------------------------------------------

@st.cache_data
def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["issued_year"].notna()].copy()
    g = (
        d.groupby("issued_year")
        .agg(
            permit_count=("permitno", "nunique"),
            total_m3_per_year=("max_m3_per_year", "sum"),
            median_m3_per_year=("max_m3_per_year", "median"),
        )
        .reset_index()
    )
    return g


@st.cache_data
def sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "purpose_category" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    g = (
        d.groupby("purpose_category")
        .agg(
            permit_count=("permitno", "nunique"),
            total_m3_per_year=("max_m3_per_year", "sum"),
            median_m3_per_year=("max_m3_per_year", "median"),
        )
        .reset_index()
    )
    total = g["total_m3_per_year"].sum()
    if total > 0:
        g["share_of_volume_%"] = g["total_m3_per_year"] / total * 100
    else:
        g["share_of_volume_%"] = np.nan
    return g


@st.cache_data
def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "surface_or_ground" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby("surface_or_ground")
        .agg(
            permit_count=("permitno", "nunique"),
            total_m3_per_year=("max_m3_per_year", "sum"),
        )
        .reset_index()
    )
    total = g["total_m3_per_year"].sum()
    if total > 0:
        g["share_of_volume_%"] = g["total_m3_per_year"] / total * 100
    else:
        g["share_of_volume_%"] = np.nan
    return g


@st.cache_data
def outlier_table(df: pd.DataFrame, q: float = 0.99) -> pd.DataFrame:
    d = df[df["max_m3_per_year"] > 0].copy()
    if d.empty:
        return d
    threshold = d["max_m3_per_year"].quantile(q)
    out = d[d["max_m3_per_year"] >= threshold].copy()
    out = out.sort_values("max_m3_per_year", ascending=False)
    return out


@st.cache_data
def municipality_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "p_municip" not in df.columns:
        return None
    d = df.copy()
    g = (
        d.groupby("p_municip")
        .agg(
            permit_count=("permitno", "nunique"),
            total_m3_per_year=("max_m3_per_year", "sum"),
        )
        .reset_index()
    )
    return g


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def format_m3(x: float) -> str:
    if x is None or np.isnan(x):
        return "–"
    if x >= 1e9:
        return f"{x/1e9:,.2f} B"
    if x >= 1e6:
        return f"{x/1e6:,.2f} M"
    if x >= 1e3:
        return f"{x/1e3:,.2f} K"
    return f"{x:,.0f}"


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------

st.sidebar.title("💧 PTTW Analytics Dashboard")
st.sidebar.markdown(
    """
Explore Ontario’s **Permit-to-Take-Water (PTTW)** data by time, sector,
source type, geography, and outliers.
"""
)

page = st.sidebar.radio(
    "Navigate to:",
    [
        "Overview",
        "Sector Explorer",
        "Surface vs Groundwater",
        "Spatial Explorer",
        "Outlier Explorer",
        "Data & Methodology",
    ],
)


# ---------------------------------------------------------------------
# PAGE 1 – OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.title("Ontario Permit-to-Take-Water Analytics – Overview")

    st.markdown(
        "High-level view of **how much water is permitted** and "
        "**how that has changed over time**."
    )

    ysum = yearly_summary(pttw)
    ssum = sector_summary(pttw)

    # headline metrics
    base = pttw[pttw["issued_year"].notna()]
    total_permits = base["permitno"].nunique()
    total_volume = base["max_m3_per_year"].sum()

    latest_year = int(base["issued_year"].max())
    latest = base[base["issued_year"] == latest_year]
    latest_volume = latest["max_m3_per_year"].sum()

    prev_year = latest_year - 1
    prev = base[base["issued_year"] == prev_year]
    prev_volume = prev["max_m3_per_year"].sum() if not prev.empty else np.nan
    yoy_change = (
        (latest_volume - prev_volume) / prev_volume * 100
        if prev_volume and prev_volume > 0
        else np.nan
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total permits (all years)", f"{total_permits:,}")
    c2.metric("Permitted volume (all years, m³/year)", format_m3(total_volume))
    c3.metric(
        f"{latest_year} total permitted volume (m³/year)",
        format_m3(latest_volume),
        f"{yoy_change:+.1f}% vs {prev_year}" if not np.isnan(yoy_change) else None,
    )

    st.markdown("---")

    # time series
    if not ysum.empty:
        min_year = int(ysum["issued_year"].min())
        max_year = int(ysum["issued_year"].max())
        year_range = st.slider(
            "Year range for trends:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
        )
        ys = ysum[
            (ysum["issued_year"] >= year_range[0])
            & (ysum["issued_year"] <= year_range[1])
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=ys["issued_year"],
                y=ys["permit_count"],
                name="Number of permits",
                opacity=0.6,
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ys["issued_year"],
                y=ys["total_m3_per_year"],
                name="Total m³/year",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Permits and total permitted volume over time",
            xaxis_title="Year",
            yaxis=dict(title="Permits", side="left"),
            yaxis2=dict(
                title="Total m³/year",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=40, t=50, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    # sector bar
    st.markdown("### Top sectors by permitted volume")
    if not ssum.empty:
        top_n = st.slider("Show top N sectors:", 3, 15, 8)
        s = ssum.sort_values("total_m3_per_year", ascending=False).head(top_n)
        fig2 = px.bar(
            s,
            x="purpose_category",
            y="total_m3_per_year",
            text="share_of_volume_%",
            labels={
                "purpose_category": "Sector",
                "total_m3_per_year": "Total m³/year",
                "share_of_volume_%": "Share of volume (%)",
            },
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            xaxis_tickangle=45,
            margin=dict(l=40, r=40, t=40, b=120),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 2 – SECTOR EXPLORER
# ---------------------------------------------------------------------
elif page == "Sector Explorer":
    st.title("Sector Explorer")
    st.markdown(
        "Compare **sectors** over time and inspect their volume distributions."
    )

    ssum = sector_summary(pttw)
    if ssum.empty:
        st.warning("No `purpose_category` column found – cannot build sector view.")
    else:
        all_sectors = sorted(ssum["purpose_category"].tolist())
        default_sectors = all_sectors[:4] if len(all_sectors) >= 4 else all_sectors

        selected = st.multiselect(
            "Select sectors:",
            options=all_sectors,
            default=default_sectors,
        )

        df = pttw[pttw["purpose_category"].isin(selected)].copy()
        df = df[df["issued_year"].notna()]

        if df.empty:
            st.info("No data for the selected sectors.")
        else:
            grp = (
                df.groupby(["issued_year", "purpose_category"])
                .agg(total_m3_per_year=("max_m3_per_year", "sum"))
                .reset_index()
            )
            fig = px.line(
                grp,
                x="issued_year",
                y="total_m3_per_year",
                color="purpose_category",
                markers=True,
                labels={
                    "issued_year": "Year",
                    "total_m3_per_year": "Total m³/year",
                    "purpose_category": "Sector",
                },
            )
            fig.update_layout(title="Total permitted volume over time by sector")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Distribution of permitted volumes (log scale)")
            d_box = df[df["max_m3_per_year"] > 0]
            fig_box = px.box(
                d_box,
                x="purpose_category",
                y="max_m3_per_year",
                color="purpose_category",
                points=False,
                labels={
                    "purpose_category": "Sector",
                    "max_m3_per_year": "Max m³/year per permit",
                },
            )
            fig_box.update_yaxes(type="log")
            fig_box.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 3 – SURFACE vs GROUNDWATER
# ---------------------------------------------------------------------
elif page == "Surface vs Groundwater":
    st.title("Surface vs Groundwater")
    st.markdown(
        "Compare **surface water** and **groundwater** in terms of permits and volume."
    )

    ssum = source_summary(pttw)
    if ssum.empty:
        st.warning("No `surface_or_ground` column found – cannot build this view.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                ssum,
                names="surface_or_ground",
                values="total_m3_per_year",
                title="Share of total permitted volume",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_bar = px.bar(
                ssum,
                x="surface_or_ground",
                y="permit_count",
                labels={
                    "surface_or_ground": "Source type",
                    "permit_count": "Number of permits",
                },
                title="Number of permits by source type",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        df = pttw[pttw["issued_year"].notna()].copy()
        grp = (
            df.groupby(["issued_year", "surface_or_ground"])
            .agg(total_m3_per_year=("max_m3_per_year", "sum"))
            .reset_index()
        )
        fig_ts = px.line(
            grp,
            x="issued_year",
            y="total_m3_per_year",
            color="surface_or_ground",
            markers=True,
            labels={
                "issued_year": "Year",
                "total_m3_per_year": "Total m³/year",
                "surface_or_ground": "Source type",
            },
            title="Total permitted volume over time by source type",
        )
        st.plotly_chart(fig_ts, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 4 – SPATIAL EXPLORER
# ---------------------------------------------------------------------
elif page == "Spatial Explorer":
    st.title("Spatial Explorer")
    st.markdown(
        "Interactive map of **high-volume permits**, with filters for year and volume."
    )

    if not {"latitude", "longitude"}.issubset(pttw.columns):
        st.warning("Latitude/longitude columns not found in the dataset.")
    else:
        geo = pttw.dropna(subset=["latitude", "longitude"]).copy()
        geo = geo[
            (geo["latitude"].between(40, 57))
            & (geo["longitude"].between(-96, -74))
        ]
        geo = geo[geo["max_m3_per_year"] > 0]

        min_year = int(geo["issued_year"].min())
        max_year = int(geo["issued_year"].max())

        c1, c2 = st.columns(2)
        with c1:
            year_min, year_max = st.slider(
                "Year range:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1,
            )
        with c2:
            volume_q = st.slider(
                "Minimum volume (quantile on max m³/year per permit):",
                0.0, 0.99, 0.8, 0.01,
            )

        threshold = geo["max_m3_per_year"].quantile(volume_q)
        geo_filt = geo[
            (geo["issued_year"] >= year_min)
            & (geo["issued_year"] <= year_max)
            & (geo["max_m3_per_year"] >= threshold)
        ]

        if geo_filt.empty:
            st.info("No permits match the current filters.")
        else:
            fig_map = px.scatter_mapbox(
                geo_filt,
                lat="latitude",
                lon="longitude",
                color="purpose_category",
                size="max_m3_per_year",
                hover_data={
                    "permitno": True,
                    "purpose_category": True,
                    "specific_purpose": True,
                    "max_m3_per_year": ":,.0f",
                    "issued_year": True,
                },
                zoom=4,
                height=550,
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", y=-0.1),
                title="High-volume permits (circle size = max m³/year)",
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # simple municipality ranking if available
        msum = municipality_summary(pttw)
        if msum is not None and not msum.empty:
            st.markdown("### Top municipalities by total permitted volume")
            top_m = (
                msum.sort_values("total_m3_per_year", ascending=False)
                .head(15)
                .copy()
            )
            top_m["total_m3_per_year_fmt"] = top_m["total_m3_per_year"].apply(format_m3)
            st.dataframe(
                top_m[["p_municip", "total_m3_per_year_fmt", "permit_count"]]
                .rename(
                    columns={
                        "p_municip": "Municipality",
                        "total_m3_per_year_fmt": "Total m³/year",
                        "permit_count": "Permits",
                    }
                ),
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# PAGE 5 – OUTLIER EXPLORER
# ---------------------------------------------------------------------
elif page == "Outlier Explorer":
    st.title("Outlier Explorer")
    st.markdown("Focus on the **largest individual permits** in the dataset.")

    out = outlier_table(pttw, q=0.99)
    if out.empty:
        st.info("No outliers found (check that `max_m3_per_year` exists and is > 0).")
    else:
        sectors = sorted(out["purpose_category"].dropna().unique().tolist()) \
            if "purpose_category" in out.columns else []
        sources = sorted(out["surface_or_ground"].dropna().unique().tolist()) \
            if "surface_or_ground" in out.columns else []

        c1, c2, c3 = st.columns(3)
        with c1:
            sec_filter = st.multiselect(
                "Sector filter:",
                options=sectors,
                default=[],
            )
        with c2:
            src_filter = st.multiselect(
                "Source filter:",
                options=sources,
                default=[],
            )
        with c3:
            year_min = int(out["issued_date"].dt.year.min())
            year_max = int(out["issued_date"].dt.year.max())
            year_range = st.slider(
                "Issue year range:",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
            )

        mask = out["issued_date"].dt.year.between(year_range[0], year_range[1])
        if sec_filter:
            mask &= out["purpose_category"].isin(sec_filter)
        if src_filter:
            mask &= out["surface_or_ground"].isin(src_filter)

        out_f = out[mask].copy().sort_values("max_m3_per_year", ascending=False)
        out_f["max_m3_per_year_fmt"] = out_f["max_m3_per_year"].apply(format_m3)

        st.markdown("### Largest permits (filtered)")
        cols_to_show = [
            "permitno",
            "purpose_category",
            "specific_purpose",
            "surface_or_ground",
            "max_m3_per_year_fmt",
            "issued_date",
            "expiry_date",
        ]
        cols_present = [c for c in cols_to_show if c in out_f.columns]
        st.dataframe(
            out_f[cols_present].rename(
                columns={
                    "permitno": "Permit ID",
                    "purpose_category": "Sector",
                    "specific_purpose": "Specific purpose",
                    "surface_or_ground": "Source",
                    "max_m3_per_year_fmt": "Max m³/year",
                    "issued_date": "Issued",
                    "expiry_date": "Expiry",
                }
            ),
            use_container_width=True,
            height=450,
        )

        if "permit_duration_years" in out_f.columns:
            st.markdown("### Volume vs permit duration (log scale on volume)")
            fig_sc = px.scatter(
                out_f,
                x="permit_duration_years",
                y="max_m3_per_year",
                color="purpose_category" if "purpose_category" in out_f.columns else None,
                hover_data=["permitno", "surface_or_ground"] if "surface_or_ground" in out_f.columns else ["permitno"],
                labels={
                    "permit_duration_years": "Permit duration (years)",
                    "max_m3_per_year": "Max m³/year",
                },
            )
            fig_sc.update_yaxes(type="log")
            st.plotly_chart(fig_sc, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 6 – DATA & METHODOLOGY
# ---------------------------------------------------------------------
else:
    st.title("Data & Methodology")

    st.markdown(
        """
**What this dashboard does**

- Ingests Ontario’s PTTW permit data.
- Cleans and standardises key fields (dates, sectors, source types, volumes).
- Derives annual volumes, permit durations, and simple flags for surface vs groundwater.
- Builds time-series, sector profiles, spatial maps, and outlier views.

        """
    )
