import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PMEGP Executive Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 20px;}
    h1 {color: #1f77b4; text-align: center; padding: 20px;}
    h2 {color: #2c3e50; border-bottom: 2px solid #1f77b4; padding-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(csv_path):
    """Load PMEGP dataset with actual fields"""
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        # Ensure "SECTOR" exists based on ACTIVITY_NAME
        manufacturing_activities = [
            "Manufacturing", "Food Processing", "Textile Unit", "Handicraft"
        ]
        if "SECTOR" not in df.columns:
            df["SECTOR"] = df["ACTIVITY_NAME"].apply(
                lambda x: "Manufacturing" if x in manufacturing_activities else "Services"
            )
        num_cols = [
            'EMPLOYMENT_AT_SETUP',
            'PROJ_COST',
            'MARGIN_MONEY_SUBSIDY_RS',
            'ANNUAL_TURNOVER',
            'MM_REL_AMT'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        date_cols = ['DOB', 'BANK_F_DATE', 'MM_CLAIM_DT', 'MM_REL_DT', 'EDP_CERT_DT']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'PROJ_COST' in df.columns and 'MM_REL_AMT' in df.columns:
            df['ROI_PERCENT'] = ((df['ANNUAL_TURNOVER'] - df['PROJ_COST']) /
                                     (df['PROJ_COST'] + 1) * 100).fillna(0)
        if 'MM_CLAIM_DT' in df.columns:
            df['MONTH_YEAR'] = df['MM_CLAIM_DT'].dt.strftime('%Y-%m')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def apply_filters(df, filters):
    filtered_df = df.copy()
    if filters['states'] and len(filters['states']) > 0:
        filtered_df = filtered_df[filtered_df['STATE_NM'].isin(filters['states'])]
    if filters['activities'] and len(filters['activities']) > 0:
        filtered_df = filtered_df[filtered_df['ACTIVITY_NAME'].isin(filters['activities'])]
    if filters['categories'] and len(filters['categories']) > 0:
        filtered_df = filtered_df[filtered_df['BENF_CATEGORY_DESC'].isin(filters['categories'])]
    if filters['status'] and len(filters['status']) > 0:
        filtered_df = filtered_df[filtered_df['CURRENT_STATUS'].isin(filters['status'])]
    if filters['location'] and len(filters['location']) > 0:
        filtered_df = filtered_df[filtered_df['IND_TYPE'].isin(filters['location'])]
    if 'sectors' in filters and filters['sectors'] and len(filters['sectors']) > 0:
        filtered_df = filtered_df[filtered_df['SECTOR'].isin(filters['sectors'])]
    return filtered_df

def calculate_metrics(df):
    if len(df) == 0:
        return None
    return {
        'Total_Enterprises': len(df),
        'Total_Employment': int(df['EMPLOYMENT_AT_SETUP'].sum()),
        'Total_Project_Cost_Cr': round(df['PROJ_COST'].sum() / 10000000, 2),
        'Total_Subsidy_Cr': round(df['MARGIN_MONEY_SUBSIDY_RS'].sum() / 10000000, 2),
        'Operational_Rate': round((df['OPERATIONAL_STATUS'] == 'Operational').sum() / len(df) * 100, 2),
        'Female_Rate': round((df['GENDER'] == 'Female').sum() / len(df) * 100, 2),
        'Avg_ROI': round(df['ROI_PERCENT'].mean(), 2) if 'ROI_PERCENT' in df.columns else 0,
    }

def create_kpi_cards(metrics):
    if metrics is None:
        st.warning("No data available")
        return
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Enterprises", f"{metrics['Total_Enterprises']:,}")
    with col2:
        st.metric("Employment", f"{metrics['Total_Employment']:,}")
    with col3:
        st.metric("Project Cost (Cr)", f"₹{metrics['Total_Project_Cost_Cr']:.2f}")
    with col4:
        st.metric("Subsidy (Cr)", f"₹{metrics['Total_Subsidy_Cr']:.2f}")
    with col5:
        st.metric("Operational %", f"{metrics['Operational_Rate']:.1f}%")
    with col6:
        st.metric("Female %", f"{metrics['Female_Rate']:.1f}%")

def create_state_performance_map(df):
    if len(df) == 0:
        return None
    state_coords = {
        'HIMACHAL PRADESH': (31.7724, 77.1025),
        'UTTAR PRADESH': (26.8467, 80.9462),
        'MAHARASHTRA': (19.7515, 75.7139),
        'KARNATAKA': (15.3173, 75.7139),
        'TAMIL NADU': (11.1271, 79.2787),
        'RAJASTHAN': (27.0238, 74.2179),
        'WEST BENGAL': (24.8355, 88.2676),
        'BIHAR': (25.0961, 85.3131),
        'PUNJAB': (31.1471, 75.3412),
        'HARYANA': (29.0588, 77.0745),
    }
    state_data = df.groupby('STATE_NM').agg({
        'EMPLOYMENT_AT_SETUP': 'sum',
        'APP_ID': 'count',
        'MARGIN_MONEY_SUBSIDY_RS': 'sum',
        'OPERATIONAL_STATUS': lambda x: (x == 'Operational').sum(),
        'PROJ_COST': 'mean',
        'ANNUAL_TURNOVER': 'mean'
    }).reset_index()
    state_data.columns = ['state', 'employment', 'enterprises', 'subsidy',
                          'operational', 'avg_cost', 'avg_turnover']
    state_data['operational_rate'] = (state_data['operational'] / state_data['enterprises'] * 100).round(1)

    max_employment = state_data['employment'].max()
    m = folium.Map(location=[22.5, 78], zoom_start=4, tiles='OpenStreetMap')
    for _, row in state_data.iterrows():
        state = row['state']
        if state in state_coords:
            coords = state_coords[state]
            employment = int(row['employment'])
            enterprises = int(row['enterprises'])
            operational_pct = row['operational_rate']
            subsidy = int(row['subsidy'])
            intensity = (employment / max_employment) * 0.8 + 0.2
            color_val = int(intensity * 255)
            color = f'#{color_val:02x}86FF'
            radius = max(30000, min(150000, enterprises * 5000)
            )
            popup_html = f"""
            <div style="font-family: Arial; width: 300px;">
                <h3 style="color: #1f77b4;"><b>{state}</b></h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Enterprises:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>{enterprises:,}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Employment:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>{employment:,}</b></td>
                    </tr>
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Operational:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>{operational_pct:.1f}%</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Subsidy (Cr):</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>₹{subsidy/10000000:.2f}</b></td>
                    </tr>
                </table>
            </div>
            """
            folium.Circle(
                location=coords,
                radius=radius,
                popup=folium.Popup(popup_html, max_width=350),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2,
                tooltip=f"{state}: {enterprises} enterprises, {employment:,} jobs"
            ).add_to(m)
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 250px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; color: #1f77b4;">Map Legend</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;">
    • Circle Size = Enterprises<br>
    • Color Intensity = Employment<br>
    • Darker Blue = More Jobs<br>
    • Click for Details
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def create_sector_analysis(df):
    if len(df) == 0 or 'SECTOR' not in df.columns or 'ROI_PERCENT' not in df.columns:
        return None
    sector_data = df.groupby('SECTOR').agg(
        Employment=('EMPLOYMENT_AT_SETUP', 'sum'),
        ROI=('ROI_PERCENT', 'mean')
    ).reset_index()
    sector_data = sector_data.sort_values('Employment', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sector_data['SECTOR'],
        x=sector_data['Employment'],
        orientation='h',
        marker=dict(
            color=sector_data['ROI'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="ROI %")
        ),
        text=[f"{int(v):,} jobs" for v in sector_data['Employment']],
        textposition='outside'
    ))
    fig.update_layout(
        title='<b>Sector-wise Employment Generation (Color = ROI %)</b>',
        xaxis_title='Employment',
        yaxis_title='Sector',
        height=400,
        template='plotly_white',
        margin=dict(l=120)
    )
    return fig

def create_activity_analysis(df):
    """Analyze enterprises by activity type"""
    if len(df) == 0:
        return None
    activity_data = df.groupby('ACTIVITY_NAME').agg({
        'APP_ID': 'count',
        'EMPLOYMENT_AT_SETUP': 'sum',
        'PROJ_COST': 'mean',
        'MARGIN_MONEY_SUBSIDY_RS': 'sum',
        'OPERATIONAL_STATUS': lambda x: (x == 'Operational').sum()
    }).reset_index()
    activity_data['operational_rate'] = (activity_data['OPERATIONAL_STATUS'] /
                                         activity_data['APP_ID'] * 100)
    activity_data = activity_data.sort_values('EMPLOYMENT_AT_SETUP', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=activity_data['ACTIVITY_NAME'],
        x=activity_data['EMPLOYMENT_AT_SETUP'],
        orientation='h',
        marker=dict(
            color=activity_data['operational_rate'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Operational %"),
        ),
        text=[f"{emp:,.0f} jobs" for emp in activity_data['EMPLOYMENT_AT_SETUP']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Employment: %{x:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title='<b>Activity-wise Employment Generation</b>',
        xaxis_title='Employment',
        yaxis_title='Activity Type',
        height=500,
        template='plotly_white',
        margin=dict(l=200)
    )
    return fig

def create_financial_analysis(df):
    """Financial metrics analysis"""
    if len(df) == 0:
        st.warning("No data")
        return
    
    num_cols = [
        'EMPLOYMENT_AT_SETUP',
        'PROJ_COST',
        'MARGIN_MONEY_SUBSIDY_RS',
        'ANNUAL_TURNOVER',
        'MM_REL_AMT'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        if 'ROI_PERCENT' in df.columns:
            fig.add_trace(go.Histogram(
                x=df[df['ROI_PERCENT'] < 500]['ROI_PERCENT'],
                nbinsx=40,
                marker=dict(color='#2E86AB'),
                hovertemplate='ROI: %{x:.0f}%<br>Count: %{y}<extra></extra>'
            ))
            fig.update_layout(
                title='<b>ROI Distribution</b>',
                xaxis_title='ROI %',
                yaxis_title='Enterprises',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ROI_PERCENT not found in data, cannot plot ROI Distribution.")
            
    with col2:
        required = ['PROJ_COST', 'MARGIN_MONEY_SUBSIDY_RS', 'ANNUAL_TURNOVER']
        if all(col in df.columns for col in required):
            activity_financial = df.groupby('ACTIVITY_NAME').agg({
                'PROJ_COST': 'mean',
                'MARGIN_MONEY_SUBSIDY_RS': 'mean',
                'ANNUAL_TURNOVER': 'mean'
            }).reset_index()
            for c in required:
                activity_financial[c] = pd.to_numeric(activity_financial[c], errors='coerce')
                activity_financial[c] /= 100000
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=activity_financial['ACTIVITY_NAME'],
                y=activity_financial['PROJ_COST'],
                name='Avg Cost',
                marker=dict(color='#FF6B6B')
            ))
            fig.add_trace(go.Bar(
                x=activity_financial['ACTIVITY_NAME'],
                y=activity_financial['MARGIN_MONEY_SUBSIDY_RS'],
                name='Avg Subsidy',
                marker=dict(color='#4ECDC4')
            ))
            fig.add_trace(go.Bar(
                x=activity_financial['ACTIVITY_NAME'],
                y=activity_financial['ANNUAL_TURNOVER'],
                name='Avg Turnover',
                marker=dict(color='#06A77D')
            ))
            fig.update_layout(
                title='<b>Activity-wise Financial Metrics</b>',
                xaxis_title='Activity',
                yaxis_title='Amount (Lakhs)',
                height=400,
                barmode='group',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns for Activity-wise Financial Metrics missing.")
            
def create_demographics(df):
    """Gender and category analysis"""
    if len(df) == 0:
        st.warning("No data")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_data = df['GENDER'].value_counts().reset_index()
        fig = go.Figure(data=[go.Pie(
            labels=gender_data['GENDER'],
            values=gender_data['count'],
            marker=dict(colors=['#FF6B6B', '#4ECDC4']),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
        )])
        fig.update_layout(title='<b>Gender Distribution</b>', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        category_data = df['BENF_CATEGORY_DESC'].value_counts().head(6).reset_index()
        fig = go.Figure(data=[go.Pie(
            labels=category_data['BENF_CATEGORY_DESC'],
            values=category_data['count'],
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
        )])
        fig.update_layout(title='<b>Category Distribution</b>', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        location_data = df['IND_TYPE'].value_counts().reset_index()
        fig = go.Figure(data=[go.Pie(
            labels=location_data['IND_TYPE'],
            values=location_data['count'],
            marker=dict(colors=['#06A77D', '#F18F01']),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
        )])
        fig.update_layout(title='<b>Rural vs Urban</b>', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
def create_clustering_analysis(df):
    """Enterprise clustering with profiling"""
    if len(df) < 10:
        st.warning("Not enough data for clustering")
        return
    try:
        cluster_features = ['EMPLOYMENT_AT_SETUP', 'PROJ_COST', 'SUSTAINABILITY_SCORE']
        X = df[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        cluster_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': clusters.astype(str),
            'Employment': df['EMPLOYMENT_AT_SETUP'].values,
            'ProjectCost': df['PROJ_COST'].values,
            'Sustainability': df['SUSTAINABILITY_SCORE'].values,
            'Status': df['OPERATIONAL_STATUS'].values
        })
        cluster_descriptions = {
            '0': 'Struggling/At-Risk',
            '1': 'Developing/Moderate',
            '2': 'Growing/Good',
            '3': 'High-Performer'
        }
        cluster_colors = {
            '0': '#FF8C42',
            '1': '#FF6B6B',
            '2': '#4ECDC4',
            '3': '#06A77D'
        }
        fig = go.Figure()
        for cluster_id in sorted(cluster_df['Cluster'].unique()):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
            cluster_count = len(cluster_data)
            cluster_pct = (cluster_count / len(cluster_df)) * 100
            fig.add_trace(go.Scatter(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                mode='markers',
                name=f"Cluster {cluster_id}: {cluster_descriptions[cluster_id]} ({cluster_count:,} | {cluster_pct:.1f}%)",
                marker=dict(
                    size=8,
                    color=cluster_colors[cluster_id],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[
                    f"<b>Cluster {cluster_id}: {cluster_descriptions[cluster_id]}</b><br>" +
                    f"Employment: {emp:,.0f}<br>" +
                    f"Project Cost: ₹{cost:,.0f}<br>" +
                    f"Sustainability: {sus:.1f}%<br>" +
                    f"Status: {status}"
                    for emp, cost, sus, status in zip(
                        cluster_data['Employment'],
                        cluster_data['ProjectCost'],
                        cluster_data['Sustainability'],
                        cluster_data['Status']
                    )
                ],
                hovertemplate='%{text}<extra></extra>'
            ))
        fig.update_layout(
            title={
                'text': '<b>Enterprise Clustering Analysis (PCA)</b><br>' +
                        '<sub>Grouped by Employment, Cost & Sustainability</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=f'<b>PC1 ({pca.explained_variance_ratio_[0]:.1%})</b>',
            yaxis_title=f'<b>PC2 ({pca.explained_variance_ratio_[1]:.1%})</b>',
            height=600,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        # Cluster summary
        st.markdown("### Cluster Summary")
        summary_data = []
        for cluster_id in sorted(cluster_df['Cluster'].unique()):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
            count = len(cluster_data)
            pct = (count / len(cluster_df)) * 100
            summary_data.append({
                'Cluster': f"{cluster_id}: {cluster_descriptions[cluster_id]}",
                'Count': f"{count:,}",
                '%': f"{pct:.1f}%",
                'Avg Employment': f"{cluster_data['Employment'].mean():.0f}",
                'Avg Cost (₹L)': f"{cluster_data['ProjectCost'].mean()/100000:.2f}",
                'Sustainability': f"{cluster_data['Sustainability'].mean():.1f}%"
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Clustering error: {e}")
    

def main():
    csv_path = '/Users/apple/Documents/0. MSME project/V2_PMEGP_Dashboard/data/PMEGP_Generated_Data_2023-26.csv'
    st.title("PMEGP Executive Dashboard")
    st.markdown("*Real-time PMEGP Monitoring System with Actual Field Structure*")
    st.markdown("---")

    df = load_data(csv_path)
    if df is None:
        st.sidebar.error("CSV file not found")
        st.error(f"Please generate data first using: python generate_pmegp_data.py")
        return

    st.sidebar.success("✓ Data loaded successfully")
    st.sidebar.info(f"Records: {len(df)}")

    st.sidebar.header("Filters")
    filters = {
        'states': st.sidebar.multiselect('States', sorted(df['STATE_NM'].unique()), default=sorted(df['STATE_NM'].unique())[:5]),
        'activities': st.sidebar.multiselect('Activities', sorted(df['ACTIVITY_NAME'].unique()), default=sorted(df['ACTIVITY_NAME'].unique())[:3]),
        'categories': st.sidebar.multiselect('Category', sorted(df['BENF_CATEGORY_DESC'].unique()), default=sorted(df['BENF_CATEGORY_DESC'].unique())),
        'status': st.sidebar.multiselect('Status', sorted(df['CURRENT_STATUS'].unique()), default=sorted(df['CURRENT_STATUS'].unique())),
        'location': st.sidebar.multiselect('Location', sorted(df['IND_TYPE'].unique()), default=sorted(df['IND_TYPE'].unique())),
        'sectors': st.sidebar.multiselect('Sector', sorted(df['SECTOR'].unique()), default=sorted(df['SECTOR'].unique()))
    }

    filtered_df = apply_filters(df, filters)
    st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)}")

    metrics = calculate_metrics(filtered_df)
    st.header("Key Performance Indicators")
    create_kpi_cards(metrics)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Geographic", "Activities", "Demographics",
        "Financial", "Clustering", "Data"
    ])

    with tab1:
        st.header("Geographic Analysis")
        st.info("**How to read**: Larger circles = more enterprises. Darker blue = more employment. Click for details.")
        map_obj = create_state_performance_map(filtered_df)
        if map_obj:
            st_folium(map_obj, width=1400, height=600)

    with tab2:
        st.header("Activity Analysis")
        sector_chart = create_sector_analysis(filtered_df)
        if sector_chart:
            st.plotly_chart(sector_chart, use_container_width=True)
        chart = create_activity_analysis(filtered_df)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    with tab3:
        st.header("Demographics")
        create_demographics(filtered_df)

    with tab4:
        st.header("Financial Analysis")
        create_financial_analysis(filtered_df)

    with tab5:
        st.header("Clustering Analysis")
        create_clustering_analysis(filtered_df)

    with tab6:
        st.header("Data View")
        if len(filtered_df) > 0:
            st.dataframe(filtered_df.head(100), use_container_width=True)
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download CSV", csv,
                              f"PMEGP_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>PMEGP Dashboard | Ministry of MSME</div>",
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    