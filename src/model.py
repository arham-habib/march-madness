"""
TODO
- Add Rishabhs viz
- Add RFE pipeline
- show the params for logistic regresison
- Run inference on only the last row before march madness
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import optuna
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

class BaseModel:

    def __init__(self, data, unpacked_data, validation_data, features, target, random_state=42):
        """
        Initialize the base model with data and parameters.

        Parameters:
            data (pd.DataFrame): The full dataset.
            unpacked_data (pd.DataFrame): The dataset unpacked to a non-versus format
            validation_data (pd.DataFrame): the dataset to validate on
            features (list): List of feature column names.
            target (str): Target column name.
            random_state (int): Random seed for reproducibility.
        """
        self.data = data.copy()
        self.unpacked_data = unpacked_data
        self.validation_data = validation_data
        self.features = features
        self.target = target
        self.random_state = random_state
        self.model = None
        self.split_data()

    def split_data(self):
        """
        Split data into train, test, and validation sets
        Train and test are before March 10th of the last year (80/20 split),
        validation is on or after March 10th.
        """
        self.compute_last_observable_features()

        game_cols = ['Season', 'DayNum', 'home_team', 'away_team', 'home_score', 'away_score']
        validation_data = self.validation_data.loc[:, game_cols].copy()

        home_features = self.last_observable_features.add_suffix('_home')
        away_features = self.last_observable_features.add_suffix('_away')

        validation_data = (
            validation_data
            .merge(home_features, left_on='home_team', right_on='team_home', how='left')
            .merge(away_features, left_on='away_team', right_on='team_away', how='left')
        )

        validation_data.drop(columns=['team_home', 'team_away'], inplace=True)
        pre_cutoff_data = self.data
        
        # Randomly split pre-cutoff data into train and test sets
        train_size = 0.8 # TODO MAKE HYPERPARAM
        train_indices = np.random.RandomState(self.random_state).rand(len(pre_cutoff_data)) < train_size
        
        self.train_data = self.add_matchup_features(pre_cutoff_data[train_indices].copy())
        self.test_data = self.add_matchup_features(pre_cutoff_data[~train_indices].copy())
        self.validation_data = self.add_matchup_features(validation_data)
        
        # Set exponential weights with half-life of 6000 games
        decay_rate = np.log(2) / 6000  # Calculate decay rate from half-life
        time_points = np.arange(len(self.train_data))
        self.weights = np.exp(-decay_rate * time_points)

        self.drop_nans()

    def drop_nans(self):
        """Drop rows with NaN values in feature columns for all datasets."""
        self.train_data = self.train_data.dropna(subset=self.features)
        self.test_data = self.test_data.dropna(subset=self.features) 
        self.validation_data = self.validation_data.dropna(subset=self.features)

    def compute_last_observable_features(self):
        """
        Compute and store the last observable features for each team from self.data
        prior to the cutoff date. This method stacks the home and away records,
        sorts them by date, and then groups by team to extract the most recent record.
        
        Parameters:
            cutoff (str or datetime): The cutoff date before which to consider games.
            
        Returns:
            pd.DataFrame: A dataframe containing the last observable features for each team.
        """
        pre_cutoff_data = (
            self.unpacked_data # [self.unpacked_data[self.date_column] < cutoff]
            .sort_values(['Season', 'DayNum'])
            .groupby('team')
            .last()
            .reset_index()
        )
        self.last_observable_features = pre_cutoff_data

    def add_matchup_features(self, df):
        """
        Add matchup features by computing the differences between home and away exponential moving averages (EMAs),
        and add a column indicating if the home team won.

        Parameters:
            df (pd.DataFrame): DataFrame containing the raw feature columns.

        Returns:
            pd.DataFrame: A new DataFrame with additional matchup feature columns.
        """
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Compute the difference in EMAs for various metrics
        df["points_ema_diff"] = df["ema_points_home"] - df["ema_points_away"]
        df["offensive_rating_ema_diff"] = df["ema_offensive_rating_home"] - df["ema_defensive_rating_away"]
        df["defensive_rating_ema_diff"] = df["ema_defensive_rating_home"] - df["ema_offensive_rating_away"]
        df["2_pt_pct_ema_diff"] = df["ema_2p_pct_home"] - df["ema_2p_pct_away"]
        df["3_pt_pct_ema_diff"] = df["ema_3p_pct_home"] - df["ema_3p_pct_away"]
        df["ft_pct_ema_diff"] = df["ema_ft_pct_home"] - df["ema_ft_pct_away"]
        df["off_rebound_rate_ema_diff"] = df["ema_off_rebound_rate_home"] - df["ema_def_rebound_rate_away"]
        df["def_rebound_rate_ema_diff"] = df["ema_def_rebound_rate_home"] - df["ema_off_rebound_rate_away"]
        df["turnover_rate_ema_diff"] = df["ema_turnover_rate_away"] - df["ema_turnover_rate_home"]
        df["assist_turnover_ema_diff"] = df["ema_assist_turnover_home"] - df["ema_assist_turnover_away"]
        df["block_rate_ema_diff"] = df["ema_block_rate_home"] - df["ema_block_rate_away"]
        df["lagged_elo_diff"] = df['lagged_elo_home'] - df['lagged_elo_away'] 

        df['home_win'] = df['home_score'] > df['away_score']
        
        return df

    def visualize_calibration(self):
        """
        Plot calibration curves for train, test, and validation sets.

        Returns:
            tuple: (fig, axes) where fig is the matplotlib figure and axes is an array of subplot axes.
        """
        # Create a figure with three subplots side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Define datasets to plot
        datasets = [
            (self.train_data, "Train"),
            (self.test_data, "Test"),
            (self.validation_data, "Validation")
        ]
        
        # Iterate over each dataset and corresponding axis
        for ax, (dataset, title) in zip(axes, datasets):
            # Extract features and target
            X = dataset[self.features]
            y = dataset[self.target]
            
            # Get predicted probabilities using the subclass's predict_proba method
            prob_pred = self.predict_proba(X)
            
            # Compute calibration curve with 20 bins (approximately 5% increments)
            fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pred, n_bins=20)
            
            # Plot the calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            
            # Plot the perfectly calibrated line
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            
            # Set titles and labels
            ax.set_title(f"{title} Calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig, axes

    def train(self):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    def predict_proba(self, X):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    def visualize_rolling_log_loss(self, window_size=100):
        """
        Plot rolling log loss over time for train, test, and validation sets.

        Parameters:
            window_size (int): Size of the rolling window.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, dataset, title in zip(
            axes, [self.train_data, self.test_data, self.validation_data], ["Train", "Test", "Validation"]
        ):
            df = dataset.sort_values(by=["Season", "DayNum"])
            X = df[self.features]
            y = df[self.target]
            preds = self.predict_proba(X)
            rolling_losses = [
                log_loss(y.iloc[i - window_size:i], preds[i - window_size:i], labels=[0, 1])
                for i in range(window_size, len(df))
            ]
            indices = np.arange(len(rolling_losses))
            ax.plot(indices, rolling_losses)
            ax.set_title(f"{title} Rolling Log Loss")
            ax.set_xlabel("Date")
            ax.set_ylabel("Log Loss")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        return fig, ax

    def visualize_calibration(self):
        """Plot calibration curves for train, test, and validation sets."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, dataset, title in zip(
            axes, [self.train_data, self.test_data, self.validation_data], ["Train", "Test", "Validation"]
        ):
            X = dataset[self.features]
            y = dataset[self.target]
            preds = self.predict_proba(X)
            frac_pos, mean_pred = calibration_curve(y, preds, n_bins=20)
            ax.plot(mean_pred, frac_pos, marker="o", label=title)
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_title(f"{title} Calibration")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Probability")
            ax.legend()
        return fig, ax

    def show_confusion_matrix(self, threshold=0.5):
        """
        Display confusion matrices for train, test, and validation sets.

        Parameters:
            threshold (float): Probability threshold for classification.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, dataset, title in zip(
            axes, [self.train_data, self.test_data, self.validation_data], ["Train", "Test", "Validation"]
        ):
            X = dataset[self.features]
            y = dataset[self.target]
            preds = self.predict_proba(X) > threshold
            cm = confusion_matrix(y, preds)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax)
            ax.set_title(f"{title} Confusion Matrix")
        return fig, ax


class XGBoostModel(BaseModel):
    def __init__(self, data, unpacked_data, validation_data, features, target, optuna_n_trials=50, random_state=42):
        """
        Initialize the XGBoost model.

        Parameters:
            data (pd.DataFrame): The full dataset.
            unpacked_data (pd.DataFrame)
            features (list): List of feature column names.
            target (str): Target column name.
            optuna_n_trials (int): Number of Optuna trials for tuning.
            random_state (int): Random seed for reproducibility.
        """
        super().__init__(data, unpacked_data, validation_data, features, target, random_state)
        self.optuna_n_trials = optuna_n_trials
        self.best_params = None
        # Preprocess data to ensure correct types at initialization
        self._preprocess_data()

    def _preprocess_data(self):
        """Ensure features and target are of type float64 in train, test, and validation data."""
        # Convert features and target to float64
        self.train_data[self.features] = self.train_data[self.features].astype(np.float64)
        self.train_data[self.target] = self.train_data[self.target].astype(np.float64)
        self.test_data[self.features] = self.test_data[self.features].astype(np.float64)
        self.test_data[self.target] = self.test_data[self.target].astype(np.float64)
        # If validation_data exists (depends on BaseModel)
        if hasattr(self, 'validation_data'):
            self.validation_data[self.features] = self.validation_data[self.features].astype(np.float64)
            self.validation_data[self.target] = self.validation_data[self.target].astype(np.float64)

    def objective(self, trial):
        """Define the Optuna objective function for hyperparameter tuning."""
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": self.random_state,
        }
        model = XGBClassifier(**params)
        # Ensure data types before fitting
        X_train = self.train_data[self.features].astype(np.float64)
        y_train = self.train_data[self.target].astype(np.float64)
        model.fit( X_train, y_train)
        X_test = self.test_data[self.features].astype(np.float64)
        y_test = self.test_data[self.target].astype(np.float64)
        preds = model.predict_proba(X_test)[:, 1].astype(np.float64)  # Ensure predictions are float64
        return log_loss(y_test, preds)

    def train(self):
        """Train the XGBoost model with Optuna tuning (without Platt scaling)."""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.optuna_n_trials)
        self.best_params = study.best_params
        params = self.best_params.copy()
        params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": self.random_state})
        model = XGBClassifier(**params)
        # Ensure data types before fitting
        X_train = self.train_data[self.features].astype(np.float64)
        y_train = self.train_data[self.target].astype(np.float64)
        model.fit(X_train, y_train)
        # Set the model directly without Platt scaling
        self.model = model

    def predict_proba(self, X):
        """
        Predict probabilities using the trained XGBoost model.

        Parameters:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        # Ensure input X is float64
        if isinstance(X, pd.DataFrame):
            X = X[self.features].astype(np.float64).values
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float64)
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    def visualize_features(self):
        """Visualize feature importance using SHAP values."""
        # Use the XGBoost model directly for SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.train_data[self.features].astype(np.float64))
        shap.summary_plot(shap_values, self.train_data[self.features])


class LogisticRegressionModel(BaseModel):
    def __init__(self, data, unpacked_data, features, target, date_column, optuna_n_trials=50, random_state=42):
        """
        Initialize the Logistic Regression model.

        Parameters:
            data (pd.DataFrame): The full dataset.
            unpacked_data (pd.DataFrame)
            features (list): List of feature column names.
            target (str): Target column name.
            date_column (str): Column name with datetime information.
            optuna_n_trials (int): Number of Optuna trials for tuning.
            random_state (int): Random seed for reproducibility.
        """
        super().__init__(data, unpacked_data, features, target, date_column, random_state)
        self.optuna_n_trials = optuna_n_trials
        self.best_params = None
        self.scaler = StandardScaler()
        # Scale features after initial split
        self.split_data_with_scaling()

    def split_data_with_scaling(self):
        """Override split_data to include feature scaling."""
        super().split_data()
        self.train_data[self.features] = self.scaler.fit_transform(self.train_data[self.features])
        self.test_data[self.features] = self.scaler.transform(self.test_data[self.features])
        self.validation_data[self.features] = self.scaler.transform(self.validation_data[self.features])

    def objective(self, trial):
        """Define the Optuna objective function for hyperparameter tuning."""
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=self.random_state)
        model.fit(self.train_data[self.features], self.train_data[self.target])
        preds = model.predict_proba(self.test_data[self.features])[:, 1]
        return log_loss(self.test_data[self.target], preds)

    def train(self):
        """Train the Logistic Regression model with Optuna tuning."""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.optuna_n_trials)
        self.best_params = study.best_params
        model = LogisticRegression(
            C=self.best_params["C"], penalty="l2", solver="lbfgs", max_iter=1000, random_state=self.random_state
        )
        model.fit(self.train_data[self.features], self.train_data[self.target])
        self.model = model
        
        X_train_sm = self.train_data[self.features]
        logit_model = sm.Logit(self.train_data[self.target], X_train_sm)
        result = logit_model.fit(disp=False)

        print("Logistic Regression Summary (without intercept):")
        print(result.summary())

    def predict_proba(self, X):
        """
        Predict probabilities using the trained model.

        Parameters:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        X_scaled = self.scaler.transform(X[self.features])
        return self.model.predict_proba(X_scaled)

    def visualize_features(self):
        """Visualize feature importance using model coefficients."""
        coef = self.model.coef_[0]
        feature_names = self.features
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names, coef)
        ax.set_xlabel("Coefficient")
        ax.set_title("Feature Importance (Coefficients)")
        plt.tight_layout()
        return fig, ax  # Return fig and ax instead of plt.show()
    