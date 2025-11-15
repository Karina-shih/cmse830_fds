import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from ucimlrepo import fetch_ucirepo
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import io
# We no longer need 'missingno'
# import missingno as msno 

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Liver Disease Dataset Analyzer")

# --- Data Loading and Preprocessing (Cached for Performance) ---

@st.cache_data
def load_hepatitis_data():
    """Load and clean the Hepatitis dataset (UCI ID 46)."""
    try:
        hepatitis = fetch_ucirepo(id=46)
    except Exception as e:
        st.error(f"Failed to load Hepatitis dataset: {e}")
        return None, None, None, None

    X = hepatitis.data.features
    y = hepatitis.data.targets
    df_raw = pd.concat([X, y], axis=1) # This is the raw data we need
    df_clean = df_raw.copy()

    # 1. Calculate and record initial missing values (for reporting)
    missing_data_info = df_raw.isnull().sum()
    missing_data_percentage = (missing_data_info / len(df_raw)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data_info,
        'Percentage (%)': missing_data_percentage.round(2)
    })
    
    # 2. Data Cleaning and Imputation (based on the provided notebook logic)
    mode_cols = ['Steroid', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 'Liver Firm', 
                 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices']
    for col in mode_cols:
        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 1.0
        df_clean[col] = df_clean[col].fillna(mode_val)
    
    df_clean['Bilirubin'] = df_clean['Bilirubin'].fillna(df_clean['Bilirubin'].median())
    df_clean['Sgot'] = df_clean['Sgot'].fillna(df_clean['Sgot'].median())

    mice_numeric_cols = ['Alk Phosphate', 'Albumin', 'Protime']
    for col in mice_numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if df_clean[mice_numeric_cols].isnull().any().any():
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_clean[mice_numeric_cols] = imputer.fit_transform(df_clean[mice_numeric_cols])

    for col in mode_cols:
        df_clean[col] = df_clean[col].astype(int)
        
    # Rename 'Class' for better display (1: Dead, 2: Live, as per screenshot)
    df_clean['Class_Label'] = df_clean['Class'].map({1: '1 (Dead)', 2: '2 (Live)'}).astype('category')
        
    buffer = io.StringIO()
    df_clean.info(buf=buffer)
    info_str = buffer.getvalue()
    buffer.close()

    return df_clean, df_raw, missing_df, info_str

@st.cache_data
def load_cirrhosis_data():
    """Load and clean the Cirrhosis dataset (UCI ID 878)."""
    try:
        cirrhosis = fetch_ucirepo(id=878)
    except Exception as e:
        st.error(f"Failed to load Cirrhosis dataset: {e}")
        return None, None, None, None 
    
    X = cirrhosis.data.features
    y = cirrhosis.data.targets
    df_raw = pd.concat([X, y], axis=1) # This is the raw data
    df_clean = df_raw.copy()

    # 1. Calculate and record initial missing values (for reporting)
    missing_data_info = df_raw.isnull().sum()
    missing_data_percentage = (missing_data_info / len(df_raw)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data_info,
        'Percentage (%)': missing_data_percentage.round(2)
    })

    # 2. Data Cleaning and Imputation (based on the provided notebook logic)
    df_clean['Platelets'] = (
        df_clean['Platelets']
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace(['?', '-', 'NA', 'N/A', '', 'NaNN', 'NaN'], np.nan) 
        .astype(float)
    )
    df_clean['Platelets'] = df_clean['Platelets'].fillna(0).astype('int64')

    mode_cols = ['Stage', 'Drug', 'Ascites', 'Hepatomegaly', 'Spiders']
    for col in mode_cols:
        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'N'
        df_clean[col] = df_clean[col].fillna(mode_val)

    median_val = df_clean['Prothrombin'].median()
    df_clean['Prothrombin'] = df_clean['Prothrombin'].fillna(median_val)
    
    mapping_YN = {'Y': 1, 'N': 0}
    df_clean['Ascites'] = df_clean['Ascites'].map(mapping_YN).fillna(0).astype(int) 
    df_clean['Hepatomegaly'] = df_clean['Hepatomegaly'].map(mapping_YN).fillna(0).astype(int) 
    df_clean['Spiders'] = df_clean['Spiders'].map(mapping_YN).fillna(0).astype(int) 
    df_clean['Sex'] = df_clean['Sex'].map({'F': 1, 'M': 2}).fillna(1).astype(int) 
    df_clean['Edema'] = df_clean['Edema'].map({'N': 0, 'S': 1, 'Y': 2}).fillna(0).astype(int) 
    df_clean['Status'] = df_clean['Status'].map({'C': 1, 'D': 2, 'CL': 3}).fillna(1).astype(int) 

    mice_numeric_cols = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides']
    for col in mice_numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if df_clean[mice_numeric_cols].isnull().any().any():
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_clean[mice_numeric_cols] = imputer.fit_transform(df_clean[mice_numeric_cols])

    df_clean['Age_year'] = np.round(df_clean['Age'] / 365).astype(int)
    df_clean = df_clean.drop(columns=['Age']) 

    buffer = io.StringIO()
    df_clean.info(buf=buffer)
    info_str = buffer.getvalue()
    buffer.close()

    return df_clean, df_raw, missing_df, info_str


# Load all data
df_hepatitis_clean, df_hepatitis_raw, hepatitis_missing_df, hepatitis_info_str = load_hepatitis_data()
df_cirrhosis_clean, df_cirrhosis_raw, cirrhosis_missing_df, cirrhosis_info_str = load_cirrhosis_data()


# --- PLOTTING FUNCTION 1: Symptom Mortality Proportion (Hepatitis) ---
def plot_hepatitis_mortality_proportion(df, feature_col, ax, add_legend=False):
    """
    Generates a bar plot for the proportion of mortality (Class=1) based on a symptom (1 or 2), 
    matching the requested screenshot style.
    Class: 1=Dead, 2=Live. Symptom: 1=No, 2=Yes (typically)
    """
    if df is None or feature_col not in df.columns or 'Class' not in df.columns:
        return
        
    # Recode Class column for calculation: 1=Mortality, 0=Survival
    df_temp = df.copy()
    df_temp['Mortality'] = df_temp['Class'].apply(lambda x: 1 if x == 1 else 0)
    
    # Calculate Mean Mortality (which is the proportion of Class=1)
    # The output is P(Class=1 | Symptom)
    mortality_prop = df_temp.groupby(feature_col)['Mortality'].mean().reset_index()
    mortality_prop.columns = [feature_col, 'Proportion']
    
    # This plot uses dark_background, which is fine
    plt.style.use('dark_background') 

    # Use Matplotlib/Seaborn for the desired style
    sns.barplot(
        x=feature_col, 
        y='Proportion', 
        data=mortality_prop, 
        ax=ax, 
        color='#348ABD', # Custom Blue color
        edgecolor='black',
        linewidth=2 # Thicker border for bar
    )
    
    # Add labels on bars (centered)
    for index, row in mortality_prop.iterrows():
        ax.text(index, row['Proportion'] / 2, f"{row['Proportion']:.4f}", 
                color='black', ha="center", va="center", fontsize=12, fontweight='bold')
        
    ax.set_title(f"Proportion of Mortality in {feature_col} Symptom", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_xlabel(feature_col)
    sns.despine(ax=ax) # Clean up the chart borders

    # Manually add a legend if requested, matching the screenshot's 'Class 2' entry
    if add_legend:
        # Create a proxy artist for the legend entry 'Class 2'
        proxy_handle = plt.Rectangle((0,0),1,1, fc='#348ABD', edgecolor='black')
        ax.legend([proxy_handle], ['2'], title='Class', loc='center right', 
                   bbox_to_anchor=(1.25, 0.8), fontsize=10, title_fontsize=12)
                   
    plt.rcdefaults() # Reset after this plot


# --- PLOTTING FUNCTION 2: Age vs Status/Class Violin Plot (Hepatitis & Cirrhosis) ---
def plot_age_vs_status_violin(df, age_col, status_col, title):
    """Generates an interactive Plotly Violin Plot for Age vs Status/Class."""
    if df is None or age_col not in df.columns or status_col not in df.columns:
        st.warning(f"Required columns missing for {title} violin plot.")
        return None

    # Determine colors based on the screenshot
    if "Hepatitis" in title:
        color_map = {'1 (Dead)': '#0A4A86', '2 (Live)': '#6BB0D3'}
    elif "Cirrhosis" in title:
        color_map = {'D': '#78A9C9', 'C': '#004488', 'CL': '#F9A99D'}
    else:
        color_map = None

    fig = px.violin(
        df, 
        y=age_col, 
        x=status_col, 
        color=status_col, 
        box=True, # Display box plots inside
        points="all", # Display all points (scatter plot)
        hover_data=[age_col],
        title=f"{title}: Interactive Violin Plot of Age vs {status_col}",
        color_discrete_map=color_map,
        height=500
    )
    
    fig.update_layout(
        template="plotly_dark", # Use dark theme for dark background effect
        showlegend=True,
        xaxis_title=status_col,
        yaxis_title="Age (Years)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_traces(meanline_visible=True, 
                      marker_opacity=0.6,
                      line_width=1,
                      box_line_width=1.5,
                      box_fillcolor='rgba(255, 255, 255, 0.2)'
    )
    
    return fig


# --- DISPLAY FUNCTION 1: Data Summary ---
def display_data_summary(df, info_str, title):
    st.header(f"Summary: {title} Dataset")
    st.subheader("Data Info")
    st.code(info_str, language='text')
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))


# --- DISPLAY FUNCTION 2: Missing Values ---
# <-- THIS IS THE MODIFIED FUNCTION ---
def display_missing_values_content(df_hep_raw, hep_missing_df, df_cirr_raw, cirr_missing_df):
    """Displays both the missing value matrices and the data tables."""
    
    st.subheader("Missing Value Heatmap (Raw Data)")
    st.markdown("This plot shows the distribution of missing data (yellow) **before** any cleaning or imputation was applied.")

    col1, col2 = st.columns(2)
    
    # Use dark_background style for both plots
    plt.style.use('dark_background')
    
    with col1:
        st.markdown("**Hepatitis Dataset**")
        if df_hep_raw is not None:
            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # --- USING SEABORN.HEATMAP INSTEAD OF MISSINGNO ---
            sns.heatmap(
                df_hep_raw.isnull(), 
                ax=ax,
                cbar=False,           # Do not show the color bar
                cmap='viridis',       # Use 'viridis' (purple/yellow) colormap
                yticklabels=False     # Hide the y-axis labels (index)
            )
            ax.set_title("Hepatitis Missing Values", fontsize=14)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Values")
            
            # Display in Streamlit
            st.pyplot(fig) 
            plt.close(fig) # Close the figure
        else:
            st.warning("Hepatitis raw data not available.")
    
    with col2:
        st.markdown("**Cirrhosis Dataset**")
        if df_cirr_raw is not None:
            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 5))

            # --- USING SEABORN.HEATMAP INSTEAD OF MISSINGNO ---
            sns.heatmap(
                df_cirr_raw.isnull(), 
                ax=ax,
                cbar=False,
                cmap='viridis',
                yticklabels=False
            )
            ax.set_title("Cirrhosis Missing Values", fontsize=14)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Values")
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close(fig) # Close the figure
        else:
            st.warning("Cirrhosis raw data not available.")

    # Reset matplotlib defaults
    plt.rcdefaults() 
    
    st.markdown("---") # Add a separator

    st.subheader("Missing Value Report (Raw Data)")
    col3, col4 = st.columns(2) # Use new column variables
    with col3:
        st.markdown("**Hepatitis Dataset**")
        st.dataframe(hep_missing_df[hep_missing_df['Missing Count'] > 0])
    with col4:
        st.markdown("**Cirrhosis Dataset**")
        st.dataframe(cirr_missing_df[cirr_missing_df['Missing Count'] > 0])
# <-- END OF MODIFIED FUNCTION ---


# --- DISPLAY FUNCTION 3: Correlation Heatmaps (Interactive with Filter) ---
def display_correlation_content(df_hepatitis_clean, df_cirrhosis_clean):
    st.subheader("Interactive Correlation Heatmaps (Cleaned Data)")
    st.markdown("Use the **Sidebar Filter** to select features for the correlation matrix.")
    
    # 1. Feature Selection Logic in Sidebar
    all_hep_cols = df_hepatitis_clean.select_dtypes(include=np.number).columns.tolist() if df_hepatitis_clean is not None else []
    all_cirr_cols = df_cirrhosis_clean.select_dtypes(include=np.number).columns.tolist() if df_cirrhosis_clean is not None else []

    # Get the unique combined set of columns to offer a comprehensive filter
    all_unique_cols = sorted(list(set(all_hep_cols + all_cirr_cols)))

    # Use a specific Streamlit state key to ensure the filter only appears once per run
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Correlation Heatmap Filter")
    
    # Filter selection
    default_cols = [col for col in all_unique_cols if col in ['Age', 'Class', 'Albumin', 'Sgot', 'Prothrombin', 'Age_year', 'Status'] or 'Bilirubin' in col]
    if not default_cols and all_unique_cols:
         default_cols = all_unique_cols[:10]

    selected_cols = st.sidebar.multiselect(
        "Select Features to Display:",
        options=all_unique_cols,
        default=default_cols
    )

    if not selected_cols:
        st.warning("Please select at least one feature from the sidebar to display the heatmap.")
        return

    # 2. Plotting Function (Interactive Plotly Express)
    def plot_interactive_correlation_heatmap(df, title, selected_features):
        if df is None:
            st.warning(f"{title} data is not available.")
            return

        # Filter the DataFrame to only include selected features that exist in the DataFrame
        df_filtered_cols = [col for col in selected_features if col in df.columns]
        if not df_filtered_cols:
             st.info(f"No selected features found in the {title} dataset.")
             return
             
        df_filtered = df[df_filtered_cols]
        
        # Calculate correlation matrix
        corr = df_filtered.corr().round(2)
        
        # Use Plotly Express for interactive heatmap
        fig = px.imshow(
            corr,
            text_auto=True, # Display correlation values
            aspect="equal",
            color_continuous_scale='Viridis',
            title=title,
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(
            autosize=True,
            height=600,
            xaxis={'side': 'bottom'},
            template='plotly_dark'
        )
        
        # Fix axis labels overlap for better viewing
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)

    # 3. Display Plots
    col1, col2 = st.columns(2)
    with col1:
        plot_interactive_correlation_heatmap(df_hepatitis_clean, "Hepatitis Correlation Matrix", selected_cols)
    with col2:
        plot_interactive_correlation_heatmap(df_cirrhosis_clean, "Cirrhosis Correlation Matrix", selected_cols)


# --- DISPLAY FUNCTION 4: Dataset Comparison Content ---
def display_comparison_content(df_hepatitis_clean, df_cirrhosis_clean):
    """Content for the Dataset Comparison tab."""
    st.header("⚖️ Dataset Comparison")
    
    # ----------------------------------------------------
    # PART 1: Hepatitis Mortality Proportion Plots 
    # ----------------------------------------------------
    st.subheader("1. Hepatitis: Mortality Proportion by Key Symptoms (Class=1)")
    st.markdown("**(Class: 1=Dead, 2=Live; Symptom: 1=No, 2=Yes)**")
    
    if df_hepatitis_clean is not None:
        # This plot is NOT affected by our change, because it uses
        # 'dark_background' explicitly, which is fine.
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plt.style.use('dark_background') # Apply dark background for the look

        # Plot 1: Fatigue (No legend)
        plot_hepatitis_mortality_proportion(df_hepatitis_clean, 'Fatigue', axes[0])
        # Plot 2: Malaise (No legend)
        plot_hepatitis_mortality_proportion(df_hepatitis_clean, 'Malaise', axes[1])
        # Plot 3: Anorexia (With manual legend for "Class 2")
        plot_hepatitis_mortality_proportion(df_hepatitis_clean, 'Anorexia', axes[2], add_legend=True) 

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust tight_layout to make space for the legend on the right
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Hepatitis data required for symptom analysis is not available.")
        
    st.markdown("---")

    # ----------------------------------------------------
    # PART 2: Age Distribution Violin Plots (Interactive)
    # ----------------------------------------------------
    st.subheader("2. Age Distribution by Outcome (Interactive Violin Plots)")
    
    col1, col2 = st.columns(2)

    # Hepatitis Violin Plot
    with col1:
        if df_hepatitis_clean is not None and 'Age' in df_hepatitis_clean.columns and 'Class_Label' in df_hepatitis_clean.columns:
            fig_hep = plot_age_vs_status_violin(
                df_hepatitis_clean, 
                age_col='Age', 
                status_col='Class_Label', 
                title='Hepatitis'
            )
            if fig_hep:
                st.plotly_chart(fig_hep, use_container_width=True)
        else:
            st.warning("Hepatitis data (Age or Class) is missing for the violin plot.")

    # Cirrhosis Violin Plot
    with col2:
        if df_cirrhosis_clean is not None and 'Age_year' in df_cirrhosis_clean.columns and 'Status' in df_cirrhosis_clean.columns:
            # Map Status back to original labels for plotting consistency
            df_cirrhosis_plot = df_cirrhosis_clean.copy()
            status_map = {1: 'C', 2: 'D', 3: 'CL'} # C=Censored, D=Dead, CL=Censored at transplant
            df_cirrhosis_plot['Status_Label'] = df_cirrhosis_plot['Status'].map(status_map)

            fig_cirr = plot_age_vs_status_violin(
                df_cirrhosis_plot, 
                age_col='Age_year', 
                status_col='Status_Label',
                title='Cirrhosis'
            )
            if fig_cirr:
                st.plotly_chart(fig_cirr, use_container_width=True)
        else:
            st.warning("Cirrhosis data (Age or Status) is missing for the violin plot.")


# --- Streamlit Page Navigation Logic ---

def main():
    st.title("🏥 Liver Disease Dataset Analysis App (Streamlit Dashboard)")

    # Sidebar Menu
    menu = {
        "Dashboard Overview": "Dashboard",
        "Hepatitis Data Analysis": "Hepatitis",
        "Cirrhosis Data Analysis": "Cirrhosis",
        "EDA": "Missing & Correlation",
        "Dataset Comparison": "Comparison" 
    }
    
    # The main menu selection
    selection = st.sidebar.selectbox("Select Analysis Item", list(menu.keys()))
    page = menu[selection]

    if page == "Dashboard":
        st.header("Welcome to the Liver Disease Dataset Analysis Tool")
        st.markdown("""
        This application provides Exploratory Data Analysis (EDA) for the **Hepatitis** and **Cirrhosis** datasets.
        
        **Sidebar Control Panel Features:**
        1.  **Missing Values & Correlation:** Use the tabs to switch between the missing value report (raw data) and the interactive correlation heatmaps (cleaned data). Use the **Correlation Heatmap Filter** in the sidebar to select features.
        2.  **Individual Data Analysis Pages:** Provides `.info()`, `.describe()`, and the first 10 rows for each dataset.
        3.  **Dataset Comparison:** Allows for side-by-side comparison of key statistics and feature distributions.
        """)

    elif page == "Hepatitis":
        if df_hepatitis_clean is not None:
            display_data_summary(
                df_hepatitis_clean, 
                hepatitis_info_str, 
                "Hepatitis"
            )
        else:
            st.warning("Hepatitis dataset failed to load or is empty.")

    elif page == "Cirrhosis":
        if df_cirrhosis_clean is not None:
            display_data_summary(
                df_cirrhosis_clean, 
                cirrhosis_info_str, 
                "Cirrhosis"
            )
        else:
            st.warning("Cirrhosis dataset failed to load or is empty.")

    elif page == "Missing & Correlation":
        st.subheader("Datasets Analysis")
        tab_missing, tab_correlation = st.tabs(["📊 Missing Value Report", "📈 Correlation Heatmaps"])

        with tab_missing:
            # This function is now the simple version
            if hepatitis_missing_df is not None and cirrhosis_missing_df is not None:
                display_missing_values_content(
                    df_hepatitis_raw, 
                    hepatitis_missing_df, 
                    df_cirrhosis_raw, 
                    cirrhosis_missing_df
                )
            else:
                st.error("Missing value data is not available.")
        
        with tab_correlation:
            # This will now display the interactive heatmaps with the sidebar filter
            display_correlation_content(df_hepatitis_clean, df_cirrhosis_clean)
            
    elif page == "Comparison":
        # This function is now correctly defined above
        display_comparison_content(df_hepatitis_clean, df_cirrhosis_clean)


if __name__ == "__main__":
    # Check for necessary libraries
    try:
        import streamlit as st
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from ucimlrepo import fetch_ucirepo
        from sklearn.impute import IterativeImputer
        import plotly.figure_factory as ff
        import plotly.express as px
        # import missingno as msno # No longer needed
    except ImportError as e:
        st.error(f"Please ensure all necessary libraries are installed. Missing library: {e.name}")
        # if e.name == 'missingno':
        #      st.error("You may need to run: pip install missingno")
        st.stop()
        
    main()