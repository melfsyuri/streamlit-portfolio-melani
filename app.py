import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



# KONFIGURASI HALAMAN 

st.set_page_config(
    page_title="Melani Yuridis | Financial Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IDENTITAS / PROFIL (sesuaikan dari CV)

PROFILE = {
    "name": "Melani Sukma Yuridis",  # ganti jika ingin nama lengkap final
    "role": "Data Analyst / Data Science Portfolio",
    "short_intro": (
        "A portfolio dashboard to explore financial and stock performance insights "
        "using an integrated monthly dataset."
    ),
    "email": "melani.yuridis@gmail.com",
    "linkedin": "https://linkedin.com/melani678",
    "github": "https://github.com/melfsyuri",  
    "location": "Indonesia",
}

# Lokasi dataset lokal (langsung pakai CSV yang sudah ada)
DEFAULT_DATA_PATH = "merged_financial_stock_with_kondisi.csv"

# LOAD DATA (CACHED)

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rapikan nama kolom (jaga-jaga jika ada spasi berlebih)
    df.columns = [c.strip() for c in df.columns]

    # Parsing tanggal Year_Month
    if "Year_Month" in df.columns:
        df["Year_Month"] = pd.to_datetime(df["Year_Month"], errors="coerce")
        df = df.sort_values("Year_Month")

    # Pastikan numeric columns benar-benar numeric (jika ada string)
    numeric_candidates = [
        "Total Aset", "Total Ekuitas", "Laba Bersih", "Wadiah Deposits",
        "Non-Profit Sharing Investments", "Total Pembiayaan",
        "ROA", "ROE", "FDR", "NPF (proxy)",
        "PC1", "PC2", "Adj Close", "Return Monthly", "Volume"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# UTILITIES

def format_number(x, is_percent=False):
    if pd.isna(x):
        return "-"
    if is_percent:
        return f"{x:.2f}%"
    # format ribuan
    return f"{x:,.2f}"


def get_sidebar_navigation():
    st.sidebar.markdown("## 📂 Navigation")
    page = st.sidebar.radio(
        "Select page",
        [
            "🏠 Home",
            "👩‍💼 About Me",
            "📊 Dashboard",
            "📁 Projects",
            "📬 Contact",
        ],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Melani Portfolio")
    return page


def render_profile_card():
    st.sidebar.markdown("## 👩‍💼 Profile")
    # Jika nanti ada foto, pakai:
    # st.sidebar.image("assets/profile_pic.png", use_container_width=True)
    st.sidebar.write(f"**{PROFILE['name']}**")
    st.sidebar.write(PROFILE["role"])
    st.sidebar.write(f"📍 {PROFILE['location']}")
    st.sidebar.markdown(f"📧 {PROFILE['email']}")
    st.sidebar.markdown(f"🔗 [LinkedIn]({PROFILE['linkedin']})")
    if PROFILE.get("github"):
        st.sidebar.markdown(f"💻 [GitHub]({PROFILE['github']})")
    st.sidebar.markdown("---")


def render_dataset_status(df: pd.DataFrame, data_path: str):
    st.sidebar.markdown("## 🗂️ Dataset")
    st.sidebar.write(f"**Source:** `{data_path}`")
    st.sidebar.write(f"Rows: **{len(df):,}**")
    st.sidebar.write(f"Columns: **{df.shape[1]}**")
    if "Year_Month" in df.columns and df["Year_Month"].notna().any():
        st.sidebar.write(
            f"Period: **{df['Year_Month'].min().date()}** → **{df['Year_Month'].max().date()}**"
        )


def build_filters(df: pd.DataFrame):
    st.sidebar.markdown("## 🔍 Dashboard Filters")

    filtered = df.copy()

    # Filter tanggal
    if "Year_Month" in filtered.columns and filtered["Year_Month"].notna().any():
        min_date = filtered["Year_Month"].min().date()
        max_date = filtered["Year_Month"].max().date()

        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered["Year_Month"].dt.date >= start_date)
                & (filtered["Year_Month"].dt.date <= end_date)
            ]

    # Filter cluster (jika ada)
    if "cluster" in filtered.columns:
        cluster_options = sorted(filtered["cluster"].dropna().astype(str).unique().tolist())
        selected_clusters = st.sidebar.multiselect(
            "Cluster",
            options=cluster_options,
            default=cluster_options,
        )
        if selected_clusters:
            filtered = filtered[filtered["cluster"].astype(str).isin(selected_clusters)]

    # Filter kondisi (jika ada)
    if "kondisi" in filtered.columns:
        kondisi_options = sorted(filtered["kondisi"].dropna().astype(str).unique().tolist())
        selected_kondisi = st.sidebar.multiselect(
            "Kondisi",
            options=kondisi_options,
            default=kondisi_options,
        )
        if selected_kondisi:
            filtered = filtered[filtered["kondisi"].astype(str).isin(selected_kondisi)]

    return filtered


# HALAMAN: HOME

def page_home(df: pd.DataFrame):
    st.title("📈 Financial & Stock Portfolio Dashboard")
    st.markdown(
        f"""
        ### {PROFILE['name']}
        **{PROFILE['role']}**

        {PROFILE['short_intro']}
        """
    )

    st.info(
        "This app displays an interactive dashboard based on the merged financial-stock dataset. "
        "It is designed for stakeholder-friendly exploration without opening a notebook."
    )

    c1, c2, c3, c4 = st.columns(4)

    latest_row = df.sort_values("Year_Month").dropna(subset=["Year_Month"]).iloc[-1] if "Year_Month" in df.columns else df.iloc[-1]

    with c1:
        val = latest_row["Adj Close"] if "Adj Close" in df.columns else np.nan
        st.metric("Latest Adj Close", format_number(val))

    with c2:
        val = latest_row["Return Monthly"] if "Return Monthly" in df.columns else np.nan
        st.metric("Latest Monthly Return", format_number(val, is_percent=True))

    with c3:
        val = latest_row["ROA"] if "ROA" in df.columns else np.nan
        st.metric("Latest ROA", format_number(val, is_percent=True))

    with c4:
        val = latest_row["ROE"] if "ROE" in df.columns else np.nan
        st.metric("Latest ROE", format_number(val, is_percent=True))

    st.markdown("---")
    st.subheader("Quick Preview")
    st.dataframe(df.head(10), width="stretch")


# HALAMAN: ABOUT ME

def page_about_me():
    st.title("👩‍💼 About Me")

    st.markdown(
        f"""
        Hello, I’m **{PROFILE['name']}**.  
        This portfolio app is built to present data analysis results in a more accessible and interactive way for non-technical stakeholders.

        I use Streamlit to transform analysis outputs into a dashboard that is easier to explore, interpret, and communicate.
        """
    )

    st.markdown("### 🎯 Focus in This Portfolio")
    st.markdown(
        """
        - Financial and stock data exploration
        - Interactive KPI monitoring
        - Trend analysis over time
        - Simple segmentation insights (cluster / kondisi)
        - Stakeholder-friendly visual storytelling
        """
    )

    st.markdown("### 🛠️ Tools")
    st.markdown(
        """
        `Python` · `Pandas` · `Plotly` · `Streamlit` · `Google Colab` · `GitHub`
        """
    )

    st.markdown("### 📄 Note")
    st.caption(
        "You can further customize this section with details from your CV "
        "(education, experience, certifications, projects, and achievements)."
    )


# HALAMAN: DASHBOARD

def page_dashboard(df: pd.DataFrame):
    st.title("📊 Dashboard")
    st.caption("Interactive exploration of merged financial-stock monthly data")

    filtered_df = build_filters(df)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # ---------- KPI ----------
    st.subheader("Key Metrics")
    k1, k2, k3, k4 = st.columns(4)

    latest = filtered_df.sort_values("Year_Month").dropna(subset=["Year_Month"]).iloc[-1] if "Year_Month" in filtered_df.columns else filtered_df.iloc[-1]
    prev = (
        filtered_df.sort_values("Year_Month").dropna(subset=["Year_Month"]).iloc[-2]
        if ("Year_Month" in filtered_df.columns and len(filtered_df.dropna(subset=["Year_Month"])) > 1)
        else None
    )

    def delta_calc(col):
        if col in filtered_df.columns and prev is not None and pd.notna(latest[col]) and pd.notna(prev[col]):
            return latest[col] - prev[col]
        return None

    with k1:
        if "Adj Close" in filtered_df.columns:
            st.metric(
                "Adj Close",
                format_number(latest["Adj Close"]),
                None if delta_calc("Adj Close") is None else format_number(delta_calc("Adj Close")),
            )

    with k2:
        if "Return Monthly" in filtered_df.columns:
            st.metric(
                "Return Monthly",
                format_number(latest["Return Monthly"], is_percent=True),
                None if delta_calc("Return Monthly") is None else f"{delta_calc('Return Monthly'):.2f}%",
            )

    with k3:
        if "ROA" in filtered_df.columns:
            st.metric(
                "ROA",
                format_number(latest["ROA"], is_percent=True),
                None if delta_calc("ROA") is None else f"{delta_calc('ROA'):.2f}%",
            )

    with k4:
        if "ROE" in filtered_df.columns:
            st.metric(
                "ROE",
                format_number(latest["ROE"], is_percent=True),
                None if delta_calc("ROE") is None else f"{delta_calc('ROE'):.2f}%",
            )

    st.markdown("---")

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Trend Overview", "💰 Financial Indicators", "🧠 Cluster & Kondisi", "📋 Data Table"]
    )

    # ---------- TAB 1: Trend Overview ----------
    with tab1:
        st.subheader("Trend Overview (Time Series)")

        if "Year_Month" not in filtered_df.columns:
            st.warning("Column `Year_Month` is required for trend charts.")
        else:
            metric_options = [
                col for col in [
                    "Adj Close", "Return Monthly", "Volume",
                    "ROA", "ROE", "FDR", "NPF (proxy)",
                    "Total Aset", "Total Ekuitas", "Laba Bersih", "Total Pembiayaan"
                ] if col in filtered_df.columns
            ]

            selected_metric = st.selectbox(
                "Select metric for trend chart",
                metric_options,
                index=0 if metric_options else None,
                key="trend_metric"
            )

            if selected_metric:
                fig_trend = px.line(
                    filtered_df,
                    x="Year_Month",
                    y=selected_metric,
                    markers=True,
                    title=f"{selected_metric} Trend Over Time",
                )
                fig_trend.update_layout(height=450)
                st.plotly_chart(fig_trend, width="stretch")

            # Multi-line comparison
            st.markdown("#### Compare Multiple Metrics")
            compare_options = [c for c in ["ROA", "ROE", "FDR", "NPF (proxy)", "Return Monthly"] if c in filtered_df.columns]
            selected_compare = st.multiselect(
                "Select metrics to compare",
                options=compare_options,
                default=compare_options[:2] if len(compare_options) >= 2 else compare_options,
                key="compare_metrics"
            )

            if selected_compare:
                long_df = filtered_df[["Year_Month"] + selected_compare].melt(
                    id_vars="Year_Month",
                    var_name="Metric",
                    value_name="Value"
                )
                fig_compare = px.line(
                    long_df,
                    x="Year_Month",
                    y="Value",
                    color="Metric",
                    markers=True,
                    title="Selected Metrics Comparison"
                )
                fig_compare.update_layout(height=450)
                st.plotly_chart(fig_compare, width="stretch")

    # ---------- TAB 2: Financial Indicators ----------
    with tab2:
        st.subheader("Financial Indicators")

        col_left, col_right = st.columns(2)

        with col_left:
            if {"Total Aset", "Total Ekuitas", "Laba Bersih"}.intersection(filtered_df.columns):
                fin_cols = [c for c in ["Total Aset", "Total Ekuitas", "Laba Bersih", "Total Pembiayaan"] if c in filtered_df.columns]
                if "Year_Month" in filtered_df.columns and fin_cols:
                    fin_long = filtered_df[["Year_Month"] + fin_cols].melt(
                        id_vars="Year_Month",
                        var_name="Indicator",
                        value_name="Value"
                    )
                    fig_fin = px.line(
                        fin_long,
                        x="Year_Month",
                        y="Value",
                        color="Indicator",
                        markers=True,
                        title="Key Financial Indicators"
                    )
                    fig_fin.update_layout(height=450)
                    st.plotly_chart(fig_fin, width="stretch")

        with col_right:
            ratio_cols = [c for c in ["ROA", "ROE", "FDR", "NPF (proxy)"] if c in filtered_df.columns]
            if ratio_cols:
                selected_ratio = st.selectbox("Select ratio to inspect", ratio_cols, key="ratio_detail")
                fig_ratio = px.bar(
                    filtered_df,
                    x="Year_Month",
                    y=selected_ratio,
                    title=f"{selected_ratio} by Month"
                )
                fig_ratio.update_layout(height=450)
                st.plotly_chart(fig_ratio, width="stretch")

        st.markdown("#### Correlation Heatmap (Numeric Variables)")
        num_df = filtered_df.select_dtypes(include=[np.number]).copy()
        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.info("Not enough numeric columns to compute correlation.")

    # ---------- TAB 3: Cluster & Kondisi ----------
    with tab3:
        st.subheader("Cluster & Kondisi Analysis")

        c1, c2 = st.columns(2)

        with c1:
            if "cluster" in filtered_df.columns:
                cluster_count = filtered_df["cluster"].astype(str).value_counts().reset_index()
                cluster_count.columns = ["cluster", "count"]
                fig_cluster = px.bar(
                    cluster_count,
                    x="cluster",
                    y="count",
                    color="cluster",
                    title="Cluster Distribution"
                )
                fig_cluster.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_cluster, width="stretch")
            else:
                st.info("Column `cluster` not found.")

        with c2:
            if "kondisi" in filtered_df.columns:
                kondisi_count = filtered_df["kondisi"].astype(str).value_counts().reset_index()
                kondisi_count.columns = ["kondisi", "count"]
                fig_kondisi = px.pie(
                    kondisi_count,
                    names="kondisi",
                    values="count",
                    title="Kondisi Composition"
                )
                fig_kondisi.update_layout(height=400)
                st.plotly_chart(fig_kondisi, width="stretch")
            else:
                st.info("Column `kondisi` not found.")

        st.markdown("#### PCA Scatter (PC1 vs PC2)")
        if {"PC1", "PC2"}.issubset(filtered_df.columns):
            color_col = "kondisi" if "kondisi" in filtered_df.columns else ("cluster" if "cluster" in filtered_df.columns else None)
            fig_pca = px.scatter(
                filtered_df,
                x="PC1",
                y="PC2",
                color=color_col,
                hover_data=[c for c in ["Year_Month", "Adj Close", "ROA", "ROE"] if c in filtered_df.columns],
                title="PCA Projection"
            )
            fig_pca.update_layout(height=500)
            st.plotly_chart(fig_pca, width="stretch")
        else:
            st.info("Columns `PC1` and `PC2` are required for PCA scatter plot.")

    # ---------- TAB 4: Data Table ----------
    with tab4:
        st.subheader("Filtered Data")
        st.dataframe(filtered_df, width="stretch")

        csv_export = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download filtered data as CSV",
            data=csv_export,
            file_name="filtered_financial_stock_dashboard.csv",
            mime="text/csv",
        )


# HALAMAN: PROJECTS

def page_projects():
    st.title("📁 Projects")

    st.markdown("### Featured Project")
    st.markdown(
        """
        **Financial & Stock Performance Dashboard (Streamlit)**  
        An interactive portfolio dashboard that combines financial indicators and stock-related metrics
        to support exploratory analysis and stakeholder-friendly reporting.
        """
    )

    st.markdown("### What this project demonstrates")
    st.markdown(
        """
        - Data loading and preprocessing with Pandas
        - Interactive filtering (date, cluster, kondisi)
        - Time-series trend visualization
        - Financial KPI monitoring
        - PCA / segmentation visualization
        - Streamlit deployment-ready dashboard structure
        """
    )

    st.markdown("### Next Improvements (optional)")
    st.markdown(
        """
        - Add forecasting page (ARIMA / ETS results from your Colab)
        - Add model performance comparison section
        - Add explainability notes / business insights panel
        - Add upload option for alternative datasets
        """
    )



# HALAMAN: CONTACT

def page_contact():
    st.title("📬 Contact")

    st.markdown(
        f"""
        If you would like to discuss my portfolio, collaboration opportunities, or data projects, feel free to reach out.

        - **Email:** {PROFILE['email']}
        - **LinkedIn:** {PROFILE['linkedin']}
        """
    )

    # Form sederhana (tidak mengirim email otomatis, hanya UI)
    st.markdown("### Leave a Message")
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message", height=120)
        submitted = st.form_submit_button("Send")

        if submitted:
            if not name or not email or not message:
                st.warning("Please complete all fields.")
            else:
                st.success("Thank you! Your message has been recorded (demo form).")
                st.caption("Tip: You can connect this to email/API later if needed.")



# MAIN APP

def main():
    # Cek file dataset
    data_path = DEFAULT_DATA_PATH
    if not Path(data_path).exists():
        st.error(
            f"Dataset file not found: `{data_path}`\n\n"
            "Please place the CSV file in the same folder as app.py "
            "or update DEFAULT_DATA_PATH."
        )
        st.stop()

    df = load_data(data_path)

    # Sidebar
    render_profile_card()
    render_dataset_status(df, data_path)
    page = get_sidebar_navigation()

    # Routing
    if page == "🏠 Home":
        page_home(df)
    elif page == "👩‍💼 About Me":
        page_about_me()
    elif page == "📊 Dashboard":
        page_dashboard(df)
    elif page == "📁 Projects":
        page_projects()
    elif page == "📬 Contact":
        page_contact()


if __name__ == "__main__":
    main()