import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go

plt.style.use('ggplot')

# Define the conversion factor for Age (Days to Years)
AGE_CONVERSION_FACTOR = 365.25
# Define the specific dataset name where Age needs conversion
CIRRHOSIS_DATASET_NAME = "Cirrhosis Data" 
HEPATITIS_DATASET_NAME = "Hepatitis Data" # Assuming this name is used
INDIAN_LIVER_DATASET_NAME = "Indian Liver Patient Data" # Assuming this name is used

# --- MODIFIED DEFINITIONS FOR SCATTER PLOT LIMITS (Dynamic Mapping) ---
SCATTER_COLS_MAPPING = {
    # Note: Added 'Age' and 'gender' to allow Age exploration, which is usually the X-axis focus
    CIRRHOSIS_DATASET_NAME: [
        'Age', 'Gender', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 
        'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
    ],
    HEPATITIS_DATASET_NAME: [
        'Age', 'Gender', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime'
    ],
    INDIAN_LIVER_DATASET_NAME: [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin'
    ]
}

# --- Distribution Plot Columns (Kept consistent for Tab 1) ---
DISTRIBUTION_COLS = ["Age", "Bilirubin", "Alk Phosphate", "Sgot", "Albumin", "Gender"] 

def display_analysis():
    """Displays the Exploratory Data Analysis (EDA) section."""
    
    st.title("✏️ Data Analysis (EDA)")
    st.markdown("This section provides tools for exploring data distributions, correlations, and relationships within the selected dataset.")

    if 'datasets' not in st.session_state or not st.session_state.datasets:
        st.error("Error: Datasets are not loaded. Please go back to the Home page to check the loading status.")
        return

    datasets = st.session_state.datasets
    available_datasets = {name: df for name, df in datasets.items() if df is not None}
    dataset_names = list(available_datasets.keys())

    if not dataset_names:
        st.warning("No successfully loaded datasets available for analysis.")
        return

    # Dataset Selection
    selected_name = st.sidebar.selectbox(
        "Select Dataset for Analysis",
        dataset_names
    )
    
    # Use a copy to avoid modifying the original dataframe in session state
    df = available_datasets[selected_name].copy() 
    
    # --- START OF DATA PREPROCESSING & CLEANUP ---
    
    # 1. Convert 'Age' from days to years ONLY FOR 'Cirrhosis Data'
    if selected_name == CIRRHOSIS_DATASET_NAME and 'Age' in df.columns and pd.api.types.is_numeric_dtype(df['Age']):
        df['Age'] = df['Age'] / AGE_CONVERSION_FACTOR
        df['Age'] = df['Age'].round(2) 
    
    # 2. Robustly Rename 'sex' (case-insensitive) to 'gender'
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'sex':
            column_mapping[col] = 'Gender'

    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # 3. Map 'gender' values (1=F, 2=M) if the column exists and contains these values
    if 'Gender' in df.columns:
        if df['Gender'].dtype in ['int64', 'float64'] or df['Gender'].astype(str).str.match(r'^[12]$').any():
            df['Gender'] = df['Gender'].replace({1: 'F', 2: 'M', 1.0: 'F', 2.0: 'M'})
            if df['Gender'].dtype != 'object':
                 df['Gender'] = df['Gender'].astype('object')
            
    # --- END OF DATA PREPROCESSING & CLEANUP ---
    
    st.header(f"Exploration for: **{selected_name}**")
    
    # Tabbed Interface for Different Analyses (Now with 4 tabs)
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution Plots", "Correlation Matrix", "Relationship Scatter", "Descriptive Statistics"])

    # --- Tab 1 Logic ---
    with tab1:
        st.subheader("Distribution Plots (Histograms)")
        available_dist_cols = [col for col in DISTRIBUTION_COLS if col in df.columns]

        if not available_dist_cols:
            st.warning("None of the required columns for distribution analysis were found in the selected dataset.")
        else:
            col_to_plot = st.selectbox("Select a Column for Distribution", available_dist_cols, key=f"{selected_name}_dist_col")
            plot_title = f"Distribution of {col_to_plot}"
            is_age_converted = selected_name == CIRRHOSIS_DATASET_NAME and col_to_plot == 'Age'
            x_label = f"{col_to_plot}" if is_age_converted else col_to_plot

            if pd.api.types.is_numeric_dtype(df[col_to_plot]):
                fig = px.histogram(df, x=col_to_plot, marginal="box", title=plot_title, labels={col_to_plot: x_label}, template="plotly_white")
            else:
                value_counts_df = df[col_to_plot].value_counts().reset_index()
                value_counts_df.columns = [col_to_plot, 'count'] 
                fig = px.bar(value_counts_df, x=col_to_plot, y='count', title=plot_title, labels={col_to_plot: x_label}, template="plotly_white")
                
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2 Logic ---
    with tab2:
        st.subheader("Correlation Matrix")
        df_numerical = df.select_dtypes(include=['number'])
        if df_numerical.empty:
            st.warning("No numerical columns to calculate correlation.")
        else:
            corr_matrix = df_numerical.corr().round(4)
            continuous_cols = corr_matrix.columns.tolist() 
            z_data = corr_matrix.values
            x_labels = continuous_cols
            y_labels = continuous_cols
            
            fig_heatmap = go.Figure(data=[go.Heatmap(z=z_data, x=x_labels, y=y_labels, colorscale='Reds',
                    colorbar=dict(title=dict(text='Correlation', side='right'), len=0.8), zmin=-1, zmax=1,)])
            fig_heatmap.update_layout(font=dict(size=10), title=f"Correlation Heatmap for {selected_name} (Interactive)",
                xaxis_title="Features", yaxis_title="Features", xaxis=dict(side='bottom', tickangle=-45, constrain='domain'),
                yaxis=dict(constrain='domain'), margin=dict(l=100, r=50, t=50, b=100))
            st.plotly_chart(fig_heatmap)

    # --- Tab 3 Logic ---
    with tab3:
        st.subheader("Bivariate Relationship Scatter Plot")
        st.markdown("A **Ordinary Least Squares (OLS)** trendline is included to visualize the linear relationship between the selected variables.")
        
        all_allowed_cols = SCATTER_COLS_MAPPING.get(selected_name, [])
        available_scatter_cols = [col for col in all_allowed_cols if col in df.columns]
        available_numeric_cols = [col for col in available_scatter_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(available_numeric_cols) < 2:
            st.warning("Not enough numerical columns available in the selected list for scatter plotting.")
            return

        col_x = st.selectbox(
            "Select X-axis Variable (Numeric)", 
            available_numeric_cols, 
            index=0, 
            key=f"{selected_name}_scatter_x"
        )
        
        default_y_index = 1 if len(available_numeric_cols) > 1 and available_numeric_cols[0] == col_x else 0
        
        col_y = st.selectbox(
            "Select Y-axis Variable (Numeric)", 
            available_numeric_cols, 
            index=min(default_y_index, len(available_numeric_cols)-1), 
            key=f"{selected_name}_scatter_y"
        )
        
        if col_x != col_y:
            
            color_var = None
            if 'Gender' in df.columns and 'Gender' in available_scatter_cols:
                color_var = 'Gender'
            
            plot_trendline = "ols"
            
            is_cirrhosis = selected_name == CIRRHOSIS_DATASET_NAME
            
            x_label = f"{col_x}" if col_x == 'Age' and is_cirrhosis else col_x
            y_label = f"{col_y}" if col_y == 'Age' and is_cirrhosis else col_y
            
            fig = px.scatter(df, x=col_x, y=col_y, 
                             color=color_var, 
                             marginal_y='box', 
                             trendline=plot_trendline, 
                             title=f"Scatter Plot of {col_x} vs {col_y}",
                             labels={col_x: x_label, col_y: y_label},
                             template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("X and Y variables cannot be the same.")
            
    # --- NEW Tab 4 Logic: Descriptive Statistics ---
    with tab4:
        st.subheader("Descriptive Statistics for Numerical Features")
        st.markdown("This table summarizes the central tendency, dispersion, and shape of the dataset's numerical features.")
        
        # Calculate descriptive statistics for all numerical columns
        desc_df = df.describe().T
        
        if desc_df.empty:
            st.warning("The selected dataset contains no numerical features for descriptive statistics.")
        else:
            # Display the resulting DataFrame using Streamlit's data table
            st.dataframe(desc_df)

display_analysis()