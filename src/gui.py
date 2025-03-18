import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from model import *
from data_processing import *

# Use session state to persist model and results between reruns
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'tester' not in st.session_state:
    st.session_state.tester = None

st.title("Model Testing Dashboard")

st.sidebar.header("Upload Data & Configure Model")
year_selected = st.sidebar.select_slider(label="End Year", options=[2021, 2022, 2023, 2024])

# Add sliders for model parameters in sidebar
alpha_param = st.sidebar.slider("EMA Alpha", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
initial_elo = st.sidebar.slider("Initial Elo Rating", min_value=1000, max_value=2000, value=1500, step=100)
k_param = st.sidebar.slider("K-Factor", min_value=10, max_value=50, value=20, step=5)

box_score_list = []
schedule_list = []

for year in range(2021, year_selected + 1):  # Add +1 to include end year
    player_file = load_player_stats(f"data/box_score_men_{year}_d1.parquet")
    schedule = load_schedule(f"data/ncaab_{year}_men_d1.parquet") 
    box_score_list.append(player_file)
    schedule_list.append(schedule)

joined_box_score = pl.concat(box_score_list)
joined_schedule_list = pl.concat(schedule_list)

feature_pipeline = MarchMadnessFeaturePipeline(
    joined_box_score.collect(),
    joined_schedule_list.collect(),
    ema_alpha=alpha_param,
    initial_elo=initial_elo,
    k_factor=k_param
)

data = feature_pipeline.build_features().to_pandas()

if data is not None:
    st.write("### Data Preview")
    st.dataframe(data.tail(10))
    
    # Let user choose columns
    all_columns = data.columns.tolist()
    feature_cols = [col for col in all_columns if "lagged" in col or "diff" in col]
    features = st.sidebar.multiselect("Select feature columns", options=feature_cols, default=feature_cols[:5])
    
    non_feature_cols = [col for col in all_columns if col not in feature_cols]
    target = st.sidebar.selectbox("Select target column", options=non_feature_cols)
    date_column = st.sidebar.selectbox("Select date column", options=non_feature_cols)
    
    model_flavor = st.sidebar.radio("Select model flavor", options=["xgboost", "logistic"])
    optuna_trials = st.sidebar.slider("Optuna Trials", min_value=10, max_value=100, value=50, step=10)
        
    if st.sidebar.button("Train Model") or st.session_state.model_trained:
        if not st.session_state.model_trained:
            with st.spinner("Training model..."):
                # Initialize the appropriate model based on user selection
                if model_flavor == "xgboost":
                    tester = XGBoostModel(
                        data=data,
                        features=features,
                        target=target,
                        date_column=date_column,
                        optuna_n_trials=optuna_trials,
                        random_state=42
                    )
                elif model_flavor == "logistic":
                    tester = LogisticRegressionModel(
                        data=data,
                        features=features,
                        target=target,
                        date_column=date_column,
                        optuna_n_trials=optuna_trials,
                        random_state=42
                    )
                # Train the model (includes hyperparameter optimization)
                tester.train()
                # Store in session state
                st.session_state.tester = tester
                st.session_state.model_trained = True
                st.session_state.best_params = tester.best_params
            st.success("Model training complete!")
        
        # Use the stored model tester for visualization
        tester = st.session_state.tester
        
        st.write("**Best Hyperparameters Found:**", st.session_state.best_params)
        
        st.header("Transformed Data Tail")
        st.dataframe(tester.train_data.tail(10))

        st.header("Feature Distributions")
        # Create subplots for feature histograms
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()  # Flatten to make indexing easier
        
        for idx, feature in enumerate(features):
            # Plot histogram with KDE
            tester.train_data[feature].hist(ax=axes[idx], bins=30, density=True, alpha=0.7)
            tester.train_data[feature].plot(kind='kde', ax=axes[idx])
            
            axes[idx].set_title(feature)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')
        
        # Remove empty subplots if any
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        st.pyplot(fig)
        
        st.header("Feature Importance")
        fig, ax = tester.visualize_features()
        st.pyplot(fig)
        
        st.header("Calibration Curves")
        fig, axes = tester.visualize_calibration()
        st.pyplot(fig)
        
        st.header("Rolling Log Loss")
        fig, ax = tester.visualize_rolling_log_loss()
        st.pyplot(fig)
        
        st.header("Visualize a Validation Row")
        if not tester.validation_data.empty:
            # Get unique dates and sort them
            dates = sorted(tester.validation_data[date_column].unique())
            selected_date = st.selectbox("Select Date", options=dates, key="selected_date")
            
            # Filter games for selected date
            date_games = tester.validation_data[tester.validation_data[date_column] == selected_date]
            
            if not date_games.empty:
                st.write(f"Games on {selected_date}:")
                for idx, row in date_games.iterrows():
                    viz_df = tester.visualize_row(row)
                    st.table(viz_df)
            else:
                st.write("No games found for the selected date.")
        else:
            st.write("No validation data available based on the selected date column.")
        
        # Add button to reset model training state
        if st.button("Reset Model Training"):
            st.session_state.model_trained = False
            st.session_state.tester = None
            st.experimental_rerun()
else:
    st.write("Please select data to begin.")