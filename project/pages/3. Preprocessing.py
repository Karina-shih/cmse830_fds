import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns
# --- Import required library for KNN Imputation ---
from sklearn.impute import KNNImputer

# --- Placeholder Data Loading (Required for standalone execution) ---
@st.cache_data
def load_dummy_data():
    """Loads dummy data to simulate the datasets in session_state."""
    data = {
        'Age': np.random.randint(18, 80, 500),
        'Bilirubin': np.abs(np.random.normal(1.5, 3.0, 500)),
        'Albumin': np.abs(np.random.normal(3.5, 0.5, 500)), # Added Albumin
        'SGOT': np.abs(np.random.normal(40, 50, 500)), # Aspartate Aminotransferase
        'Alamine_Aminotransferase': np.abs(np.random.normal(30, 40, 500)), # ALT
        'Total_Bilirubin': np.abs(np.random.normal(1.0, 2.0, 500)),
        'Aspartate_Aminotransferase': np.abs(np.random.normal(30, 40, 500)), # AST
        'Gender': np.random.choice(['Male', 'Female'], 500),
        'Sgot': np.abs(np.random.normal(50, 60, 500)) 
    }
    df = pd.DataFrame(data)
    
    # Simulate more NaNs for the KNN imputation
    # We need to create a raw version before imputation for comparison
    df_raw = df.copy() 
    df_raw.loc[10:30, 'Bilirubin'] = np.nan
    df_raw.loc[50:70, 'Albumin'] = np.nan
    
    # Ensure column names match for specific datasets
    cirrhosis_df_raw = df_raw.copy().rename(columns={'SGOT': 'SGOT'})
    hepatitis_df_raw = df_raw.copy().rename(columns={'Aspartate_Aminotransferase': 'AST'})
    indian_liver_df_raw = df_raw.copy() # Keeps Total_Bilirubin, Aspartate_Aminotransferase, Alamine_Aminotransferase
    
    datasets = {
        'Cirrhosis Data': cirrhosis_df_raw,
        'Hepatitis Data': hepatitis_df_raw,
        'Indian Liver Patient Data': indian_liver_df_raw
    }
    return datasets

# Initialize session state with dummy data if not already present
if 'datasets' not in st.session_state:
    st.session_state.datasets = load_dummy_data()

# --- Preprocessing Function (Implemented KNN Imputation & Feature Engineering) ---
@st.cache_data(show_spinner=False)
def run_preprocessing(df_raw, name):
    """
    Implements KNN Imputation and Feature Engineering as requested.
    Returns the processed dataframe.
    """
    df_processed = df_raw.copy()
    
    # 1. Age Scaling Simulation (If applicable)
    if name == 'Cirrhosis Data' and 'Age' in df_processed.columns and df_processed['Age'].max() > 1000:
        df_processed['Age'] = (df_processed['Age'] / 365.25).round(1) 

    # 2. Identify numerical columns for imputation
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    
    # 3. Missing Value Imputation (KNN Method)
    # Check if there are any NaNs in numerical columns to apply imputation
    initial_nan_count = df_processed[numerical_cols].isnull().values.sum()
    if initial_nan_count > 0:
        try:
            # Initialize KNN Imputer (n_neighbors=5 is common)
            imputer = KNNImputer(n_neighbors=5)
            
            # Fit and transform only the numerical columns
            df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
            
        except Exception as e:
            # Fallback to simple mean imputation if KNN fails
            for col in numerical_cols:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # 4. Feature Engineering (Must be done AFTER imputation to ensure clean data for ratios)
    if name == 'Indian Liver Patient Data':
        # Ratio of Bilirubin and Albumin
        if 'Total_Bilirubin' in df_processed.columns and 'Albumin' in df_processed.columns:
            # Add a small epsilon to the denominator to prevent division by zero
            epsilon = 1e-6
            df_processed['Bilirubin_Albumin_Ratio'] = df_processed['Total_Bilirubin'] / (df_processed['Albumin'] + epsilon)
            
        # Ratio of Aspartate_Aminotransferase and Alamine_Aminotransferase (AST/ALT Ratio)
        if 'Alamine_Aminotransferase' in df_processed.columns and 'Aspartate_Aminotransferase' in df_processed.columns:
            # Add a small epsilon to the denominator to prevent division by zero
            df_processed['AST_ALT_Ratio'] = df_processed['Aspartate_Aminotransferase'] / (df_processed['Alamine_Aminotransferase'] + epsilon)
            
    # 5. Categorical Encoding (Placeholder)
    # OHE is skipped here for focused demo
    
    return df_processed

# --- Comparison Plot Function using KDE (Unchanged) ---
def display_imputation_comparison(df_raw, df_processed, selected_name):
    """
    Displays a comparison of a numerical feature's distribution before and after 
    KNN Imputation using a Kernel Density Estimation (KDE) Plot.
    """
    st.subheader("🧬 Distribution Comparison: Raw vs. KNN Imputed Data (KDE Plot)")
    st.markdown("Comparing the density of features that had missing values to assess the impact of imputation.")
    
    # Identify numerical columns that had NaNs
    numerical_cols = df_raw.select_dtypes(include=np.number).columns
    cols_with_nan = numerical_cols[df_raw[numerical_cols].isnull().any()].tolist()
    
    if not cols_with_nan:
        st.info(f"All numerical features in **{selected_name}** were already complete or did not require imputation.")
        return

    # Select feature to compare
    selected_feature = st.selectbox(
        "Select Numerical Feature to Compare",
        cols_with_nan,
        key=f"kde_feat_{selected_name}", # Unique key for selectbox
        help="Features listed had missing values and were imputed using KNN."
    )
    
    # Get the raw data (excluding NaNs) and processed data for the selected feature
    raw_data = df_raw[selected_feature].dropna()
    processed_data = df_processed[selected_feature]
    
    # Plotting the KDE comparison using Matplotlib/Seaborn
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # KDE Plot for Raw Data (using only non-missing values)
        sns.kdeplot(raw_data, label='Raw Data (Non-Missing)', color='blue', fill=False, linewidth=2, ax=ax)
        
        # KDE Plot for Processed Data (including imputed values)
        sns.kdeplot(processed_data, label='KNN Imputed Data', color='red', fill=True, alpha=0.3, linewidth=2, ax=ax)
        
        ax.set_title(f'KDE Comparison of {selected_feature} Distribution', fontsize=16)
        ax.set_xlabel(selected_feature, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='upper right')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Display key statistics for quantitative assessment
        st.markdown(f"**Statistics for {selected_feature}**")
        stats = pd.DataFrame({
            'Raw Mean': [raw_data.mean().round(4)],
            'Imputed Mean': [processed_data.mean().round(4)],
            'Raw Std Dev': [raw_data.std().round(4)],
            'Imputed Std Dev': [processed_data.std().round(4)],
            'Missing Count': [df_raw[selected_feature].isnull().sum()]
        }).T.rename(columns={0: 'Value'})
        st.dataframe(stats)

    except Exception as e:
        st.error(f"An error occurred while plotting the KDE comparison: {e}")


# --- Violin Plot Function (Unchanged) ---
# --- Violin Plot Function (Modified to standardize NaNs before plotting) ---
def display_violin_plot(df, selected_name):
    """
    Displays an interactive Violin Plot comparing a categorical column 
    against a numerical column.
    """
    st.subheader("🎻 Violin Plot: Distribution Comparison")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not categorical_cols or not numerical_cols:
        st.warning("The dataset lacks both categorical and numerical features required for a Violin Plot.")
        return

    col1, col2 = st.columns(2)
    
    default_x_index = categorical_cols.index('Gender') if 'Gender' in categorical_cols else 0
    selected_x_cat = col1.selectbox(
        "Choose the Categorical Feature (X-Axis)",
        categorical_cols,
        index=default_x_index,
        key=f"violin_x_{selected_name}"
    )
    
    default_y_index = numerical_cols.index('Age') if 'Age' in numerical_cols else 0
    selected_y_num = col2.selectbox(
        "Choose the Numerical Feature (Y-Axis)",
        numerical_cols,
        index=default_y_index,
        key=f"violin_y_{selected_name}"
    )
    
    try:
        # Create a copy to modify without affecting the original DataFrame
        df_plot = df.copy() 
        
        # --- NEW STEP 1: Aggressively convert non-standard missing values to NaN ---
        # If the categorical column is an object type, replace known missing strings with numpy.nan
        if df_plot[selected_x_cat].dtype == 'object':
            # Replace empty strings ('') and strings containing only spaces (' ') with NaN
            df_plot[selected_x_cat] = df_plot[selected_x_cat].replace({
                '': np.nan, 
                ' ': np.nan
            })
            # Also, attempt to replace common string representations of missing values if they exist
            df_plot[selected_x_cat] = df_plot[selected_x_cat].replace([
                'N/A', 'NA', 'None', 'NULL', 'nan'
            ], np.nan)


        # --- NEW STEP 2: Filter out rows where the selected categorical or numerical feature is NaN ---
        # This will now catch both standard np.nan and the aggressively converted NaNs
        df_plot.dropna(subset=[selected_x_cat, selected_y_num], inplace=True)
        
        if df_plot.empty:
            st.warning(f"No valid data points remain after removing all forms of missing values in '{selected_x_cat}' or '{selected_y_num}'.")
            return
            
        # Ensure the type is correctly set for plotting
        df_plot[selected_x_cat] = df_plot[selected_x_cat].astype(str)
        
        fig_violin = px.violin(
            df_plot, # Use the filtered and cleaned DataFrame
            x=selected_x_cat,
            y=selected_y_num,
            box=True,      
            points='all',  
            color=selected_x_cat, 
            title=f"Distribution of {selected_y_num} by {selected_x_cat} in {selected_name}",
        )

        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the Violin Plot: {e}")

# --- 3D Plot Function (Unchanged) ---
def display_3d_plots(datasets):
    """Displays interactive 3D scatter plots for feature interaction with selectable axes."""
    st.header("🌐 3D Feature Interaction Comparison (Processed Data)")

    datasets_to_plot = [name for name in datasets.keys() if datasets[name] is not None]
    
    if not datasets_to_plot:
        st.warning("No available datasets to plot the 3D comparisons.")
        return

    plot_tabs = st.tabs(datasets_to_plot)
    
    # Define default features for each dataset for initial load
    default_config = {
        'Cirrhosis Data': ('Age', 'Bilirubin', 'SGOT'),
        'Hepatitis Data': ('Age', 'Bilirubin', 'Sgot'),
        'Indian Liver Patient Data': ('Age', 'Bilirubin_Albumin_Ratio', 'AST_ALT_Ratio')
    }

    for i, name in enumerate(datasets_to_plot):
        df = datasets[name]
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Get default features, check if they exist, otherwise use the first 3 available
        def_x, def_y, def_z = default_config.get(name, ('Age', 'Bilirubin', numerical_cols[2] if len(numerical_cols) > 2 else numerical_cols[0]))
        
        # Adjust default index based on availability
        default_index_x = numerical_cols.index(def_x) if def_x in numerical_cols else 0
        default_index_y = numerical_cols.index(def_y) if def_y in numerical_cols else (1 if len(numerical_cols) > 1 else 0)
        default_index_z = numerical_cols.index(def_z) if def_z in numerical_cols else (2 if len(numerical_cols) > 2 else 0)

        with plot_tabs[i]:
            if len(numerical_cols) < 3:
                st.warning(f"Dataset **{name}** only has {len(numerical_cols)} numerical features. At least 3 are required for a 3D plot.")
                continue

            st.subheader(f"Select Axes for **{name}**")

            # Feature Selectors
            col_x, col_y, col_z = st.columns(3)
            
            selected_x = col_x.selectbox(
                "X-Axis Feature", 
                numerical_cols, 
                index=default_index_x,
                key=f"3d_x_{name}"
            )
            selected_y = col_y.selectbox(
                "Y-Axis Feature", 
                numerical_cols, 
                index=default_index_y,
                key=f"3d_y_{name}"
            )
            selected_z = col_z.selectbox( 
                "Z-Axis Feature", 
                numerical_cols, 
                index=default_index_z,
                key=f"3d_z_{name}"
            )
            
            # Ensure the selected features are distinct (simple check)
            if len(set([selected_x, selected_y, selected_z])) < 3:
                st.warning("Please select three distinct features for the X, Y, and Z axes.")
            else:
                try:
                    title_plot = f"3D Interaction of {selected_x}, {selected_y}, and {selected_z} in {name}"
                    
                    fig = px.scatter_3d(
                        df,
                        x=selected_x,
                        y=selected_y,
                        z=selected_z,
                        color=selected_x, # Color by X-axis feature
                        opacity=0.7,
                        title=title_plot,
                        height=600
                    )
                    fig.update_layout(
                        scene=dict(
                            xaxis_title=selected_x,
                            yaxis_title=selected_y,
                            zaxis_title=selected_z
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred while plotting the 3D chart for {name}: {e}")

# --- Feature Importance Plot Function (Mock Data) (Unchanged) ---
def display_feature_importance_plot(df, selected_name):
    """
    Displays a mock Feature Importance plot for the processed dataset 
    (simulating results from a model).
    """
    st.subheader("💡 Simulated Feature Importance (Processed Features)")
    st.markdown("This chart simulates the relative importance of numerical features after preprocessing, typically derived from a machine learning model.")
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numerical_cols:
        st.warning("No numerical features available to display importance.")
        return
    
    try:
        # 1. Simulate Importance Scores
        np.random.seed(42) 
        importance_scores = np.random.rand(len(numerical_cols))
        
        # 2. Create DataFrame for Plotting
        importance_df = pd.DataFrame({
            'Feature': numerical_cols,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=False)
        
        # 3. Create the Bar Chart using Plotly Express
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h', # Horizontal bar chart
            title=f'Simulated Feature Importance for {selected_name}',
            color='Importance', # Color based on score
            color_continuous_scale=px.colors.sequential.Sunset # Use a warm color scheme
        )
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'}) 
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Note: Scores are randomly generated for demonstration purposes in the Preprocessing step.")
        
    except Exception as e:
        st.error(f"An error occurred while plotting the Simulated Feature Importance: {e}")

# --- Parallel Coordinates Plot Function (Removed) ---


# --- Main Display Function (Removed Tab 5, Shifted Tab 6 to Tab 5) ---
def display_preprocessing():
    """Displays the Data Preprocessing section."""
    
    st.title("🔧 Data Preprocessing")
    st.markdown("This page demonstrates the data cleaning and transformation steps applied to prepare the data for modeling.")

    if 'datasets' not in st.session_state or not st.session_state.datasets:
        st.error("Error: Datasets are not loaded. Please return to the Home page to check the loading status.")
        return

    datasets_raw = st.session_state.datasets
    available_datasets_raw = {name: df for name, df in datasets_raw.items() if df is not None}
    dataset_names = list(available_datasets_raw.keys())

    if not dataset_names:
        st.warning("No successfully loaded datasets available for preprocessing.")
        return

    # 1. Run Preprocessing on all available datasets
    processed_datasets = {}
    with st.spinner('Running KNN Imputation and Feature Engineering...'):
        for name, df_raw in available_datasets_raw.items():
            processed_datasets[name] = run_preprocessing(df_raw, name)
            

    
    # 2. Select Dataset for Detailed View
    selected_name = st.selectbox(
        "Select Dataset to view Preprocessing Results",
        list(processed_datasets.keys())
    )
    
    df_raw = available_datasets_raw[selected_name]
    df_processed = processed_datasets[selected_name] 
    
    st.header(f"Preprocessing Flow: **{selected_name}**")
    
    # --- Tabs Implementation (Removed Tab 5, adjusted tabs to 5 total) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Steps Description", 
        "Imputation Comparison", 
        "Feature Distribution (Violin Plot)", 
        "3D Feature Interaction",
        "Feature Importance"     # Now Tab 5
    ])

    with tab1:
        st.subheader("💡 Preprocessing Steps (Implemented Details)")
        st.markdown("""
        **Applied Steps:**
        * **Missing Value Imputation:** Numerical NaNs are filled using the **KNN Imputer** (n=5).
        * **Feature Engineering:** Creation of the ratio of **Total Bilirubin and Albumin** and the ratio of **Aspartate\_Aminotransferase and Alamine\_Aminotransferase** (only applied to **Indian Liver Patient Data**).
        * **Categorical Encoding:** Using **One-Hot Encoding** (Step described, but not fully implemented).
        """)
        
    with tab2:
        # Show KDE Imputation Comparison
        display_imputation_comparison(df_raw, df_processed, selected_name)
    
    with tab3:
        # Violin Plot
        display_violin_plot(df_processed, selected_name)

    with tab4:
        # 3D Plots with selectable features
        display_3d_plots(processed_datasets)

    with tab5:
        # Simulated Feature Importance Plot (Shifted to Tab 5)
        display_feature_importance_plot(df_processed, selected_name)

# Execute the main function
display_preprocessing()