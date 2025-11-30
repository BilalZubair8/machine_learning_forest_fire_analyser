
import logging

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from config import config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.poly = None
        self.feature_names = []
        # encoding mappings
        self.month_mapping = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        self.day_mapping = {
            'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
            'fri': 5, 'sat': 6, 'sun': 7
        }

    def createTemporalFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from month and day using consistent encoding"""
        df = df.copy()

        # Use encoded columns if they exist, otherwise create them
        if 'month' in df.columns and 'month_encoded' not in df.columns:
            df['month_encoded'] = df['month'].map(self.month_mapping)

        if 'day' in df.columns and 'day_encoded' not in df.columns:
            df['day_encoded'] = df['day'].map(self.day_mapping)

        # Seasonal features using encoded values
        if 'month_encoded' in df.columns:
            df['season'] = df['month_encoded'].apply(self._getSeason)
            df['is_summer'] = df['month_encoded'].isin([6, 7, 8]).astype(int)
            df['is_dry_season'] = df['month_encoded'].isin([7, 8, 9]).astype(int)

        # Day of week features using encoded values
        if 'day_encoded' in df.columns:
            df['is_weekend'] = df['day_encoded'].isin([6, 7]).astype(int)  # sat=6, sun=7

        return df

    def _getSeason(self, month: int) -> int:
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn

    def createInteractionFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()

        # Fire weather interactions
        if all(col in df.columns for col in ['FFMC', 'DMC']):
            df['FFMC_DMC_interaction'] = df['FFMC'] * df['DMC']
            df['temp_RH_interaction'] = df['temp'] * df['RH']
            df['wind_DC_interaction'] = df['wind'] * df['DC']

        # Risk indices
        if all(col in df.columns for col in ['FFMC', 'DMC', 'DC']):
            df['fire_risk_index'] = (df['FFMC'] + df['DMC'] + df['DC']) / 3
            df['severity_index'] = df['ISI'] * df['temp']

        return df

    def createPolynomialFeatures(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical columns"""
        numerical_cols = config.get('preprocessing', {}).get('numerical_features', [])
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        if numerical_cols:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = self.poly.fit_transform(df[numerical_cols])
            poly_feature_names = self.poly.get_feature_names_out(numerical_cols)

            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

            # Remove original columns to avoid duplication
            df_poly = df.drop(columns=numerical_cols)
            df_poly = pd.concat([df_poly, poly_df], axis=1)

            self.feature_names = df_poly.columns.tolist()
            return df_poly

        return df

    def engineerAllFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering...")

        df_engineered = self.createTemporalFeatures(df)
        df_engineered = self.createInteractionFeatures(df_engineered)

        if config.get('feature_engineering', {}).get('polynomial_degree', 1) > 1:
            df_engineered = self.createPolynomialFeatures(
                df_engineered,
                config.get('feature_engineering', {}).get('polynomial_degree', 2)
            )

        logger.info(f"Feature engineering completed. Final shape: {df_engineered.shape}")
        return df_engineered

    def getEncodingMappings(self):
        """Return encoding mappings for use in prediction"""
        return {
            'month_mapping': self.month_mapping,
            'day_mapping': self.day_mapping
        }