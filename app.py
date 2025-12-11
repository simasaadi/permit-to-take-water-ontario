# app.py
import os
from typing import Optional, Tuple, List

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
# Data loading helpers
# ---------------------------------------------------------------------
BASE_PARQUET = "data/processed/pttw_analysis_ready.parquet"
BASE_CSV = "data/processed/pttw_analysis_ready.csv"

SUMMARY_DIR = "data/processed/summaries"
SPATIAL_DIR = "data/processed/spatial_summaries"


@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    """Load the main analysis-ready dataset."""
    if os.path.exists(BASE_PARQUET):
        df = pd.read_parquet(BASE_PARQUET)
    elif os.path.exists(BASE_CSV):
        df = pd.read_csv(
            BASE_CSV,
            parse_dates=["issued_date", "expiry_date", "renew_date", "permit_end_date"],
        )
    else:
        raise FileNotFoundError(
            "Could not find pttw_analysis_ready.* in data/processed/. "
            "Run the cleaning notebooks to generate it."
        )

    # Ensure datetime columns
    for col in ["issued_date", "expiry_date", "renew_date", "permit_end_date"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Add handy time columns if missing
    if "issued_year" not in df.columns and "issued_date" in df.columns:
        df["issued_year"] = df["issued_date"].dt.year
    if "issued_year_month" not in df.columns and "issued_date" in df.columns:
        df["issued_year_month"] = df["issued_date"].dt.to_period("M").astype(str)

    return df


def _load_indexed_summary(path: str, index_name: str) -> Optional[pd.DataFrame]:
    """Read a summary CSV that was saved with an index, and restore the index as a column."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    # Handle common 'Unnamed: 0' pattern from pandas .to_csv()
    if "Unnamed: 0" in df.columns and index_name not in df.columns:
        df = df.rename(columns={"Unnamed: 0": index_name})

    return df


@st.cache_data(show_spinner=False)
def load_summaries():
    yearly = _load_indexed_summary(
        os.path.join(SUMMARY_DIR, "yearly_summary.csv"), "issued_year"
    )
    sector = _load_indexed_summary(
        os.path.join(SUMMARY_DIR, "sector_summary.csv"), "purpose_category"
    )
    source = _load_indexed_summary(
        os.path.join(SUMMARY_DIR, "source_summary.csv"), "surface_or_ground"
    )

    outliers_path = os.path.join(SUMMARY_DIR, "largest_permits_top_outliers.csv")
    outliers = pd.read_csv(outliers_path, parse_dates=["issued_date", "expiry_date"]) \
        if os.path.exists(outliers_path) \
        else None

    return yearly, sector, source, outliers


@st.cache_data(show_spinner=False)
def load_spatial_summaries():
    grid_path = os.path.join(SPATIAL_DIR, "grid_volume_summary.csv")
    grid = pd.read_csv(grid_path) if os.path.exists(grid_path) else None

    clusters_path = os.path.join(SPATIAL_DIR, "high_volume_clusters.csv")
    clusters = pd.read_csv(clusters_path) if os.path.exists(clusters_path) else None

    municip_path = os.path.join(SPATIAL_DIR, "municipality_summary.csv")
    municip = _load_indexed_summary(municip_path, "p_municip") if os.path.exists(municip_path) else None

    return grid, clusters, municip


# Load everything once (cached) ---------------------------------------
pttw = load_base_data()
yearly_summary, sector_summary, source_summary, outliers_df = load_summaries()
grid_summary, clusters_df, municip_summary = load_spatial_summaries()


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("💧 PTTW Analytics Dashboard")
st.sidebar.markdown(
    """
This dashboard explores Ontario’s **Permit-to-Take-Water (PTTW)** dataset
from multiple angles: time, sector, source type, geography, and outliers.

Use the navigation below to move between analytical views.
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


# Small helper for nicer big numbers
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
# PAGE 1 – OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.title("Ontario Permit-to-Take-Water Analytics – Overview")

    st.markdown(
        """
This landing page gives a high-level view of **how much water is permitted**,  
**how many permits are active**, and **how patterns have changed over time**.

The numbers and charts below are calculated from the cleaned PTTW dataset used
throughout this project.
        """
    )

    # --- headline metrics ---
    latest_year = int(pttw["issued_year"].dropna().max())
    base_filtered = pttw[pttw["issued_year"].notna()]

    total_permits = base_filtered["permitno"].nunique()
    total_volume = base_filtered["max_m3_per_year"].sum()

    latest = base_filtered[base_filtered["issued_year"] == latest_year]
    latest_permits = latest["permitno"].nunique()
    latest_volume = latest["max_m3_per_year"].sum()

    prev_year = latest_year - 1
    prev = base_filtered[base_filtered["issued_year"] == prev_year]
    prev_volume = prev["max_m3_per_year"].sum() if not prev.empty else np.nan

    yoy_change = (
        (latest_volume - prev_volume) / prev_volume * 100
        if prev_volume and prev_volume > 0
        else np.nan
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total permits (all years)",
        f"{total_permits:,}",
    )
    col2.metric(
        f"Permitted volume (all years, m³/year)",
        format_m3(total_volume),
    )
    col3.metric(
        f"{latest_year} total permitted volume (m³/year)",
        format_m3(latest_volume),
        f"{yoy_change:+.1f} % vs {prev_year}" if not np.isnan(yoy_change) else None,
    )

    st.markdown("---")

    # --- time series: permits & volume ---
    if yearly_summary is not None:
        ys = yearly_summary.copy()
        ys = ys[ys["issued_year"].notna()]

        min_year = int(ys["issued_year"].min())
        max_year = int(ys["issued_year"].max())

        year_range = st.slider(
            "Select year range for trend charts:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
        )

        ys_range = ys[(ys["issued_year"] >= year_range[0]) & (ys["issued_year"] <= year_range[1])]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=ys_range["issued_year"],
                y=ys_range["permit_count"],
                name="Number of permits",
                opacity=0.6,
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ys_range["issued_year"],
                y=ys_range["total_m3_per_year"],
                name="Total permitted volume (m³/year)",
                mode="lines+markers",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Permits and total permitted volume over time",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Number of permits"),
            yaxis2=dict(
                title="Total m³/year",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=40, t=50, b=60),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top sectors by permitted volume")

    if sector_summary is not None:
        top_n = st.slider("Show top N sectors:", 3, 15, 8)
        s = sector_summary.sort_values("total_m3_per_year", ascending=False).head(top_n)

        fig2 = px.bar(
            s,
            x="purpose_category",
            y="total_m3_per_year",
            text="share_of_volume_%",
            labels={
                "purpose_category": "Purpose category",
                "total_m3_per_year": "Total m³/year",
                "share_of_volume_%": "Share of total volume (%)",
            },
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            xaxis_tickangle=45,
            title="Top sectors by total permitted volume",
            margin=dict(l=40, r=40, t=60, b=120),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
**How to read this page**

- The metrics at the top summarize the scale of the permitting system.
- The combined bar/line chart shows whether growth over time is driven by
  **more permits** or **larger volumes per permit**.
- The sector bar chart highlights which types of use dominate provincial water demand.
        """
    )


# ---------------------------------------------------------------------
# PAGE 2 – SECTOR EXPLORER
# ---------------------------------------------------------------------
elif page == "Sector Explorer":
    st.title("Sector Explorer")

    st.markdown(
        """
This page lets you drill into specific **purpose categories** and understand their
behaviour over time and in terms of volume distribution.

Use the controls below to select one or more sectors and explore how their
permitted volumes compare.
        """
    )

    if sector_summary is None:
        st.warning("Sector summary not found. Make sure sector_summary.csv exists.")
    else:
        all_sectors = list(sector_summary["purpose_category"])
        default_sectors = all_sectors[:4] if len(all_sectors) >= 4 else all_sectors

        selected = st.multiselect(
            "Select sectors to explore:",
            options=all_sectors,
            default=default_sectors,
        )

        df = pttw[pttw["purpose_category"].isin(selected)].copy()
        df = df[df["issued_year"].notna()]

        if df.empty:
            st.info("No data available for the selected sectors.")
        else:
            # Time series by sector
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
                    "purpose_category": "Purpose category",
                },
                title="Total permitted volume over time by sector",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Distribution of permitted volumes by sector (log scale)")
            df_box = df[df["max_m3_per_year"] > 0]

            fig_box = px.box(
                df_box,
                x="purpose_category",
                y="max_m3_per_year",
                color="purpose_category",
                points=False,
                labels={
                    "purpose_category": "Purpose category",
                    "max_m3_per_year": "Max m³/year per permit",
                },
            )
            fig_box.update_layout(xaxis_tickangle=45)
            fig_box.update_yaxes(type="log")
            st.plotly_chart(fig_box, use_container_width=True)

            # Summary table
            st.markdown("### Sector summary table")
            sector_stats = (
                df.groupby("purpose_category")
                .agg(
                    permits=("permitno", "nunique"),
                    total_m3_per_year=("max_m3_per_year", "sum"),
                    median_m3_per_year=("max_m3_per_year", "median"),
                    p90_m3_per_year=("max_m3_per_year", lambda x: np.quantile(x, 0.9)),
                )
                .reset_index()
            )
            sector_stats["total_m3_per_year_fmt"] = sector_stats["total_m3_per_year"].apply(format_m3)

            st.dataframe(
                sector_stats[
                    ["purpose_category", "permits", "total_m3_per_year_fmt", "median_m3_per_year", "p90_m3_per_year"]
                ].rename(
                    columns={
                        "purpose_category": "Sector",
                        "permits": "Number of permits",
                        "total_m3_per_year_fmt": "Total m³/year (formatted)",
                        "median_m3_per_year": "Median m³/year per permit",
                        "p90_m3_per_year": "90th percentile m³/year per permit",
                    }
                ),
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# PAGE 3 – SURFACE vs GROUNDWATER
# ---------------------------------------------------------------------
elif page == "Surface vs Groundwater":
    st.title("Surface vs Groundwater")

    st.markdown(
        """
This page compares **surface water** and **groundwater** permits:
how many there are, how much volume they account for, and how their
importance has evolved over time.
        """
    )

    if source_summary is None:
        st.warning("Source summary not found. Make sure source_summary.csv exists.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            fig_pie = px.pie(
                source_summary,
                names="surface_or_ground",
                values="total_m3_per_year",
                title="Share of total permitted volume by source type",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = px.bar(
                source_summary,
                x="surface_or_ground",
                y="permit_count",
                labels={
                    "surface_or_ground": "Source type",
                    "permit_count": "Number of permits",
                },
                title="Number of permits by source type",
            )
            fig_bar.update_xaxes(categoryorder="total descending")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Time series by source type
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

        st.markdown(
            """
**Interpretation**

- The pie and bar charts show the relative importance of surface vs. groundwater.
- The time series reveals whether pressures on each source type are
  **increasing**, **decreasing**, or **stable** over time.
            """
        )


# ---------------------------------------------------------------------
# PAGE 4 – SPATIAL EXPLORER
# ---------------------------------------------------------------------
elif page == "Spatial Explorer":
    st.title("Spatial Explorer")

    st.markdown(
        """
This page visualizes the spatial footprint of water-taking permits using
interactive maps. The focus is on **where** large withdrawals are concentrated
rather than on detailed cartography.

Use the controls to filter by year, sector, and minimum permitted volume.
        """
    )

    # Need coordinates
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
            volume_quantile = st.slider(
                "Minimum volume filter (quantile on max m³/year per permit):",
                min_value=0.0,
                max_value=0.99,
                value=0.8,
                step=0.01,
            )

        q_threshold = geo["max_m3_per_year"].quantile(volume_quantile)
        geo_filt = geo[
            (geo["issued_year"] >= year_min)
            & (geo["issued_year"] <= year_max)
            & (geo["max_m3_per_year"] >= q_threshold)
        ].copy()

        if geo_filt.empty:
            st.info("No permits match the current filters.")
        else:
            # Map 1: high-volume permits as points
            st.markdown("### High-volume permits (point map)")

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
                title="Location of high-volume permits (size scaled by max m³/year)",
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # Cluster overlay if available
        if clusters_df is not None and {"latitude", "longitude"}.issubset(
            clusters_df.columns
        ):
            st.markdown("### Clusters of very high-volume permits")

            fig_cluster = px.scatter_mapbox(
                clusters_df,
                lat="latitude",
                lon="longitude",
                color="cluster",
                size="max_m3_per_year",
                hover_data={
                    "permitno": True,
                    "max_m3_per_year": ":,.0f",
                },
                zoom=4,
                height=500,
            )
            fig_cluster.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", y=-0.1),
                title="K-means clusters of top 10% highest-volume permits",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        if municip_summary is not None:
            st.markdown("### Top municipalities by total permitted volume")
            top_m = (
                municip_summary.sort_values("total_m3_per_year", ascending=False)
                .head(15)
                .copy()
            )
            top_m["total_m3_per_year_fmt"] = top_m["total_m3_per_year"].apply(format_m3)

            st.dataframe(
                top_m[["p_municip", "total_m3_per_year_fmt", "permit_count"]]
                .rename(
                    columns={
                        "p_municip": "Municipality",
                        "permit_count": "Number of permits",
                        "total_m3_per_year_fmt": "Total m³/year (formatted)",
                    }
                ),
                use_container_width=True,
            )

    st.markdown(
        """
**Reading these maps**

- Larger circles indicate permits with larger **maximum annual volumes**.
- Colors distinguish purpose categories or clusters of extremely large users.
- The municipality table helps identify administrative areas with the highest
  permitted withdrawals.
        """
    )


# ---------------------------------------------------------------------
# PAGE 5 – OUTLIER EXPLORER
# ---------------------------------------------------------------------
elif page == "Outlier Explorer":
    st.title("Outlier Explorer")

    st.markdown(
        """
This page focuses on the **largest individual permits** in the dataset.
These outliers often represent disproportionate shares of total permitted volume
and may warrant closer regulatory or policy attention.
        """
    )

    if outliers_df is None:
        st.warning("Outlier table not found. Make sure largest_permits_top_outliers.csv exists.")
    else:
        df = outliers_df.copy()

        # Filters
        sectors = sorted(df["purpose_category"].dropna().unique().tolist())
        sources = sorted(df["surface_or_ground"].dropna().unique().tolist())

        c1, c2, c3 = st.columns(3)
        with c1:
            sector_filter = st.multiselect(
                "Filter by sector (optional):",
                options=sectors,
                default=[],
            )
        with c2:
            source_filter = st.multiselect(
                "Filter by source type (optional):",
                options=sources,
                default=[],
            )
        with c3:
            year_min, year_max = int(df["issued_date"].dt.year.min()), int(
                df["issued_date"].dt.year.max()
            )
            year_range = st.slider(
                "Issue year range:",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
            )

        mask = (
            (df["issued_date"].dt.year.between(year_range[0], year_range[1]))
        )
        if sector_filter:
            mask &= df["purpose_category"].isin(sector_filter)
        if source_filter:
            mask &= df["surface_or_ground"].isin(source_filter)

        df_filt = df[mask].copy().sort_values("max_m3_per_year", ascending=False)

        st.markdown("### Largest permits (filtered view)")

        df_filt["max_m3_per_year_fmt"] = df_filt["max_m3_per_year"].apply(format_m3)

        st.dataframe(
            df_filt[
                [
                    "permitno",
                    "purpose_category",
                    "specific_purpose",
                    "surface_or_ground",
                    "max_m3_per_year_fmt",
                    "issued_date",
                    "expiry_date",
                ]
            ].rename(
                columns={
                    "permitno": "Permit ID",
                    "purpose_category": "Sector",
                    "specific_purpose": "Specific purpose",
                    "surface_or_ground": "Source type",
                    "max_m3_per_year_fmt": "Max m³/year",
                    "issued_date": "Issued",
                    "expiry_date": "Expiry",
                }
            ),
            use_container_width=True,
            height=450,
        )

        st.markdown("### Volume vs permit duration (for outliers)")

        if "permit_duration_years" in pttw.columns:
            merged = df_filt.merge(
                pttw[["permitno", "permit_duration_years"]],
                on="permitno",
                how="left",
            )

            fig_scatter = px.scatter(
                merged,
                x="permit_duration_years",
                y="max_m3_per_year",
                color="purpose_category",
                hover_data=["permitno", "specific_purpose", "surface_or_ground"],
                labels={
                    "permit_duration_years": "Permit duration (years)",
                    "max_m3_per_year": "Max m³/year",
                    "purpose_category": "Sector",
                },
                title="Outlier permits – volume vs duration",
            )
            fig_scatter.update_yaxes(type="log")
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        """
**How to use this page**

- Narrow the list using sector, source type, and issue-year filters.
- Scan the table to identify permits with especially large volumes or long durations.
- The scatter plot shows whether **long-lived permits** tend to also be **high-volume**.
        """
    )


# ---------------------------------------------------------------------
# PAGE 6 – DATA & METHODOLOGY
# ---------------------------------------------------------------------
else:
    st.title("Data & Methodology")

    st.markdown(
        """
### Project overview

This project analyzes Ontario’s **Permit-to-Take-Water (PTTW)** dataset to
understand patterns in water-taking activity across **time**, **sectors**,
**source types**, and **geography**.

The analysis pipeline is organized into four main stages:

1. **Data ingestion and initial inspection**  
   - Load the raw permit dataset and review basic structure and completeness.

2. **Cleaning and feature engineering**  
   - Standardize column names and data types.  
   - Parse and validate dates (issue, expiry, renewal).  
   - Compute key metrics such as:
     - maximum litres per day  
     - annual permitted volumes (converted to m³/year)  
     - permit durations in days and years  
     - flags for surface vs. groundwater.

3. **Exploratory and spatial analysis**  
   - Time series of permit counts and volumes.  
   - Sector and specific-purpose profiles.  
   - Surface vs groundwater comparisons over time.  
   - Spatial intensity maps and clustering of high-volume permits.

4. **Dashboard and reporting outputs**  
   - Aggregated summary tables (by year, sector, source type, municipality).  
   - Outlier lists of largest permits.  
   - Map-friendly spatial summaries (grids and clusters).

---

### How to interpret volumes

All volume metrics in the dashboard represent **maximum permitted withdrawal rates**,
not necessarily observed usage. They are expressed in **cubic metres per year (m³/year)**
for comparability across permits.

Where necessary, the project converts from:

- Litres per day × days per year → annual volume (L/year)  
- L/year ÷ 1,000 → m³/year

---

### Limitations

- Spatial visualizations use permit coordinates directly and may not capture
  the full hydrologic context (e.g., aquifer boundaries, surface water routing).
- Permits with missing or obviously invalid coordinates are excluded from
  map-based views.
- The dashboard focuses on **permitted maximums**, not actual measured withdrawals.

---

### Reproducibility

The notebooks and this dashboard are designed to be reproducible:

- All intermediate datasets (cleaned tables and summaries) are saved under
  `data/processed/` and reused here.
- The dashboard code does not perform heavy transformations; it mainly reads
  prepared tables and exposes them interactively.

If you have access to the repository, you can rerun the entire pipeline by
executing the notebooks in order and then launching this Streamlit app.
        """
    )

