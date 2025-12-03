import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(
    page_title="Multi-Dataset Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading Function ---
def load_data(file_path):
    """Loads a CSV file from the specified path, handling common errors."""
    if not os.path.exists(file_path):
        return None
    try:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {e}")
        return None

# --- Load All Datasets ---
# Find Data folder
possible_data_paths = ['Data', './Data', 'project/Data']
data_path = None
for path in possible_data_paths:
    if os.path.exists(path) and os.path.isdir(path):
        data_path = path
        break

if data_path is None:
    data_path = 'Data'

data_files = {
    "Cirrhosis Data": 'cirrhosis_clenan.csv',
    "Hepatitis Data": 'hepatitis_clenan.csv',
    "Indian Liver Patient Data": 'indian_liver_patient.csv' 
}

datasets = {}
for display_name, file_name in data_files.items():
    full_path = os.path.join(data_path, file_name)
    datasets[display_name] = load_data(full_path)

# Store datasets into Streamlit Session State
if 'datasets' not in st.session_state:
    st.session_state.datasets = datasets

# --- Page Header ---
st.title("📊 Multi-Dataset Analysis Platform")
st.markdown("### Overview Dashboard")
st.markdown("Welcome! This platform provides comprehensive analysis of liver disease datasets.")

st.divider()

# --- Dataset Loading Status ---
st.subheader("📁 Dataset Status")
cols = st.columns(3)
data_loaded_count = 0
dataset_info = {}

for i, (name, df) in enumerate(st.session_state.datasets.items()):
    with cols[i]:
        if df is not None:
            st.success(f"**{name}**")
            st.metric("Rows", f"{len(df):,}", delta="✅ Loaded")
            st.metric("Columns", len(df.columns))
            data_loaded_count += 1
            dataset_info[name] = df
        else:
            st.error(f"**{name}**")
            st.metric("Status", "Failed", delta="❌")

st.divider()

# --- Quick Statistics Overview ---
if data_loaded_count > 0:
    st.subheader("📈 Quick Statistics")
    
    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = sum([len(df) for df in dataset_info.values()])
    total_features = sum([len(df.columns) for df in dataset_info.values()])
    
    with col1:
        st.metric("Total Records", f"{total_records:,}")
    with col2:
        st.metric("Total Features", f"{total_features}")
    with col3:
        st.metric("Datasets Loaded", f"{data_loaded_count}/3")
    with col4:
        avg_records = int(total_records / data_loaded_count)
        st.metric("Avg Records/Dataset", f"{avg_records:,}")
    
    st.divider()
    
    # --- Dataset Comparison ---
    st.subheader("📊 Dataset Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for record counts
        dataset_names = list(dataset_info.keys())
        record_counts = [len(df) for df in dataset_info.values()]
        
        fig_records = go.Figure(data=[
            go.Bar(x=dataset_names, y=record_counts, 
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                   text=record_counts,
                   textposition='auto')
        ])
        fig_records.update_layout(
            title="Number of Records per Dataset",
            xaxis_title="Dataset",
            yaxis_title="Number of Records",
            height=400
        )
        st.plotly_chart(fig_records, use_container_width=True)
    
    with col2:
        # Bar chart for feature counts
        feature_counts = [len(df.columns) for df in dataset_info.values()]
        
        fig_features = go.Figure(data=[
            go.Bar(x=dataset_names, y=feature_counts,
                   marker_color=['#95E1D3', '#F38181', '#EAFFD0'],
                   text=feature_counts,
                   textposition='auto')
        ])
        fig_features.update_layout(
            title="Number of Features per Dataset",
            xaxis_title="Dataset",
            yaxis_title="Number of Features",
            height=400
        )
        st.plotly_chart(fig_features, use_container_width=True)
    
    st.divider()
    
    # --- Detailed Dataset Information ---
    st.subheader("🔍 Dataset Details")
    
    for name, df in dataset_info.items():
        with st.expander(f"📋 {name} - Click to expand"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Sample Data (First 5 rows)**")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.markdown("**Data Info**")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # Missing values
                missing = df.isnull().sum().sum()
                missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
                st.write(f"**Missing Values:** {missing} ({missing_pct:.2f}%)")
                
                # Data types
                st.markdown("**Column Types:**")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count}")
            
            # Column names
            st.markdown("**Available Columns:**")
            col_display = ", ".join(df.columns.tolist())
            st.text(col_display)
    
    st.divider()
    
    # --- Missing Data Analysis ---
    st.subheader("🔍 Missing Data Analysis")
    
    missing_data = {}
    for name, df in dataset_info.items():
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            missing_data[name] = missing_count[missing_count > 0]
    
    if missing_data:
        cols = st.columns(len(missing_data))
        for idx, (name, missing) in enumerate(missing_data.items()):
            with cols[idx]:
                st.markdown(f"**{name}**")
                fig = px.bar(
                    x=missing.values, 
                    y=missing.index,
                    orientation='h',
                    labels={'x': 'Missing Count', 'y': 'Column'},
                    title=f"Missing Values"
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in any dataset!")
    
else:
    st.error("❌ No datasets were loaded successfully. Please check your data files and paths.")
    st.info("💡 Ensure your CSV files are located in the 'Data' or 'project/Data' folder.")

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Multi-Dataset Analysis Platform | Liver Disease Research</p>
    <p><small>Use the sidebar to navigate to specific analysis pages</small></p>
</div>
""", unsafe_allow_html=True)