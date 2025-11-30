import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

from config import config

logger = logging.getLogger(__name__)

class DataPreProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.smote = SMOTE(random_state=config.data_config.random_state)
        self.originalData = None
        self.cleanedData = None

    def loadAndPreProcessData(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the forest fires dataset"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Store original data for visualization
        self.originalData = df.copy()

        # Create binary target
        df['area_binary'] = (df['area'] == 'T').astype(int)

        # Generate before-cleaning visualization
        self._plotHistogramsBeforeCleaning(df)

        # Enhanced data validation
        self._validateData(df)

        # Handle missing values
        df = self._handleMissingValues(df)

        # Handle outliers
        df = self._handleOutliers(df)

        # Encode categorical variables
        df = self._encodeCategorical(df)

        # Store cleaned data for visualization
        self.cleanedData = df.copy()

        # Generate after-cleaning visualization
        self._plotHistogramsAfterCleaning(df)

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target distribution:\n{df['area_binary'].value_counts()}")

        return df

    def _handleOutliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers for visible changes in histograms"""
        df_clean = df.copy()
        numerical_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

        for col in numerical_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers before handling
                outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()

                if outliers_before > 0:
                    # Cap outliers instead of removing them
                    df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                    df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])

                    outliers_after = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    logger.info(f"Handled {outliers_before} outliers in {col} (remaining: {outliers_after})")

        return df_clean

    def _plotHistogramsBeforeCleaning(self, df: pd.DataFrame):
        """Create histogram visualization before data cleaning"""
        logger.info("Generating histogram visualization BEFORE cleaning...")

        # Select numerical columns for visualization
        numerical_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        n_cols = 4
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                # Plot histogram
                ax = axes[i]
                sns.histplot(df[col], kde=True, ax=ax, color='red', alpha=0.7, bins=15, label='Before Cleaning')
                ax.set_title(f'BEFORE Cleaning: {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

                # Add statistics
                mean_val = df[col].mean()
                std_val = df[col].std()
                ax.axvline(mean_val, color='darkred', linestyle='--', alpha=0.8, linewidth=2,
                           label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.6, label=f'±1 STD')
                ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.6)
                ax.legend(fontsize=8)

        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('DATA DISTRIBUTION - BEFORE CLEANING (Raw Data with Outliers)',
                     fontsize=16, fontweight='bold', y=0.98, color='darkred')
        plt.tight_layout()
        plt.savefig('plots/data_before_cleaning_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print data quality summary
        self._printDataQualitySummary(df, "BEFORE CLEANING")

    def _plotHistogramsAfterCleaning(self, df: pd.DataFrame):
        """Create histogram visualization after data cleaning"""
        logger.info("Generating histogram visualization AFTER cleaning...")

        # Select numerical columns for visualization
        numerical_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        n_cols = 4
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                # Plot histogram
                ax = axes[i]
                sns.histplot(df[col], kde=True, ax=ax, color='green', alpha=0.7, bins=15, label='After Cleaning')
                ax.set_title(f'AFTER Cleaning: {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

                # Add statistics
                mean_val = df[col].mean()
                std_val = df[col].std()
                ax.axvline(mean_val, color='darkgreen', linestyle='--', alpha=0.8, linewidth=2,
                           label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.6, label=f'±1 STD')
                ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.6)
                ax.legend(fontsize=8)

        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('DATA DISTRIBUTION - AFTER CLEANING (Outliers Handled)',
                     fontsize=16, fontweight='bold', y=0.98, color='darkgreen')
        plt.tight_layout()
        plt.savefig('plots/data_after_cleaning_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print data quality summary
        self._printDataQualitySummary(df, "AFTER CLEANING")

    def _printDataQualitySummary(self, df: pd.DataFrame, stage: str):
        """Print data quality summary"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"DATA QUALITY SUMMARY - {stage}")
        logger.info(f"{'=' * 60}")

        # Basic info
        logger.info(f"Dataset Shape: {df.shape}")
        logger.info(f"Total Missing Values: {df.isnull().sum().sum()}")
        logger.info(f"Duplicate Rows: {df.duplicated().sum()}")

        # Numerical columns summary
        numerical_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        logger.info(f"\nNumerical Features Summary:")
        for col in numerical_cols:
            if col in df.columns:
                logger.info(f"  {col}:")
                logger.info(f"    Mean: {df[col].mean():.2f}")
                logger.info(f"    Std:  {df[col].std():.2f}")
                logger.info(f"    Min:  {df[col].min():.2f}")
                logger.info(f"    Max:  {df[col].max():.2f}")
                logger.info(f"    Skew: {df[col].skew():.2f}")

    def _validateData(self, df: pd.DataFrame):
        """Validate data quality"""
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()

        if missing_values > 0:
            logger.warning(f"Missing values detected: {missing_values}")

        if duplicates > 0:
            logger.warning(f"Duplicate rows detected: {duplicates}")

    def _handleMissingValues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        df_clean = df.copy()

        # For numerical columns, use median imputation
        numerical_cols = config.get('preprocessing', {}).get('numerical_features', [])
        for col in numerical_cols:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                missing_count = df_clean[col].isnull().sum()
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed {missing_count} missing values in {col} with median: {median_val:.2f}")

        return df_clean

    def _encodeCategorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables with enhanced handling"""
        df_encoded = df.copy()
        categorical_cols = config.get('preprocessing', {}).get('categorical_features', [])

        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        return df_encoded

    def prepareFeaturesTarget(self, df: pd.DataFrame):
        """Prepare features and target with validation"""
        numerical_features = config.get('preprocessing', {}).get('numerical_features', [])
        categorical_features = config.get('preprocessing', {}).get('categorical_features', [])

        feature_columns = numerical_features + [col + '_encoded' for col in categorical_features]

        # Validate all features exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        X = df[feature_columns]
        y = df['area_binary']

        logger.info(f"Features used: {feature_columns}")
        logger.info(f"Feature matrix shape: {X.shape}")

        return X, y, feature_columns

    def splitAndScaleData(self, X: pd.DataFrame, y: pd.Series, handle_imbalance: bool = True):
        """Split data and handle class imbalance"""
        # Initial split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=config.data_config.test_size,
            random_state=config.data_config.random_state,
            stratify=y
        )

        # Second split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=config.data_config.val_size,
            random_state=config.data_config.random_state,
            stratify=y_train_val
        )

        # Handle class imbalance
        if handle_imbalance:
            X_train, y_train = self.smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Training set class distribution: {np.bincount(y_train)}")

        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Data splitting and scaling completed")
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return (X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test, self.scaler)


def loadAndPreProcessData(file_path: str):
    processor = DataPreProcessor()
    return processor.loadAndPreProcessData(file_path), processor.label_encoders


def prepareFeaturesTarget(df: pd.DataFrame):
    processor = DataPreProcessor()
    return processor.prepareFeaturesTarget(df)


def splitAndScaleData(X: pd.DataFrame, y: pd.Series):
    processor = DataPreProcessor()
    return processor.splitAndScaleData(X, y)
