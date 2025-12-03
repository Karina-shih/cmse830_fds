import streamlit as st
import pandas as pd
import numpy as np # 導入 numpy 方便後續可能的操作，雖然這次主要用 pandas

def display_overview():
    """Displays the Summary and Overview of all loaded datasets."""
    
    st.title("⚙️ Data Overview")
    st.markdown("This page provides summary statistics, the first few rows, and missing value profiles for each dataset.")
    st.info("🥼This project is used to predict the probability of Liver Disease.")

    if 'datasets' not in st.session_state or not st.session_state.datasets:
        st.error("Error: Datasets have not been loaded. Please return to the Home page to check the dataset loading status.")
        return

    # 複製 session_state 中的資料集，以避免直接修改原始資料
    datasets = {name: df.copy() if df is not None else None 
                for name, df in st.session_state.datasets.items()}
    
    dataset_names = [name for name, df in datasets.items() if df is not None]

    if not dataset_names:
        st.warning("No successfully loaded datasets available to display.")
        return

    # --- 核心修改部分：針對 Cirrhosis Data 進行 Age_year 轉換 ---
    if 'Cirrhosis Data' in datasets and datasets['Cirrhosis Data'] is not None:
        df_cirrhosis = datasets['Cirrhosis Data']
        
        # 假設 'Age' 欄位是以『天』為單位
        if 'Age' in df_cirrhosis.columns:
            # 轉換為年並新增 Age_year 欄位，使用 365.25 考慮閏年
            df_cirrhosis['Age_year'] = df_cirrhosis['Age'] / 365.25
            
            # 您可以選擇將 Age_year 移到 Age 旁邊，使其更易於比較（可選）
            cols = df_cirrhosis.columns.tolist()
            age_index = cols.index('Age')
            cols.insert(age_index + 1, cols.pop(cols.index('Age_year')))
            df_cirrhosis = df_cirrhosis[cols]
            
            # 將修改後的資料集存回 datasets 字典中
            datasets['Cirrhosis Data'] = df_cirrhosis
            
    # --- 核心修改部分結束 ---


    tabs = st.tabs(dataset_names)

    for i, name in enumerate(dataset_names):
        df = datasets[name]
        
        with tabs[i]:
            st.header(f"📈 {name} - Summary Information")

            # Basic Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Rows", f"{len(df):,}")
            # 注意：在 Cirrhosis Data 的 tab 中，這裡會顯示 20 欄（19 欄 + Age_year）
            col2.metric("Number of Columns", f"{df.shape[1]:,}") 
            
            # Missing values overview
            missing_values = df.isnull().sum()
            total_missing = missing_values.sum()
            col3.metric("Total Missing Values", f"{total_missing:,}")
            
            # Data Head
            st.subheader("Dataset Head (First 5 Rows)")
            # 這裡會呈現包含 Age_year 的新資料框
            st.dataframe(df.head()) 

            # Statistical Summary 
            st.subheader("Numerical Feature Statistical Summary")
            try:
                # 統計摘要也會包含 Age_year 的 mean, std, min, max 等資訊
                st.dataframe(df.describe().T) 
            except Exception:
                st.warning("Could not generate numerical summary statistics.")
                
            # Missing values detailed list
            if total_missing > 0:
                st.subheader("Detailed Missing Values")
                missing_df = pd.DataFrame({
                    'Missing Count': missing_values,
                    'Missing Rate (%)': (missing_values / len(df)) * 100
                }).sort_values(by='Missing Count', ascending=False)
                
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])
            else:
                st.info("Because I cleaned the datast in midterm project, this dataset is clean now.")

# 確保在 Streamlit 運行時調用函數
if 'datasets' not in st.session_state:
    # 這裡只是為了讓代碼在沒有 session_state 的環境下也能運行，
    # 實際應用中，您的 main 應用應該已經載入資料到 session_state
    st.session_state.datasets = {
        'Cirrhosis Data': pd.read_csv('your_cirrhosis_data.csv') # 請替換為您的實際載入代碼
    }

display_overview()