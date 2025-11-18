"""
Property Price Prediction System
================================
Монголын үл хөдлөх хөрөнгийн үнийг таамаглах систем

This comprehensive script handles the complete pipeline:
1. Data loading from multiple CSV files
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Model training and evaluation
6. Price prediction

Features:
- Checkpoint system for resumable execution
- Progress tracking with tqdm
- Error handling and logging
- Automated model selection
- Comprehensive reporting

Author: Claude AI
Date: 2025-11-18
"""

import os
import sys
import json
import pickle
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/property_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PropertyPricePrediction:
    """
    Main class for property price prediction pipeline

    Энэ класс нь үл хөдлөх хөрөнгийн үнийг таамаглах бүх процессыг удирдана:
    - Дата ачаалах, цэвэрлэх
    - Шинжилгээ хийх
    - Модель сургах
    - Үнэ таамаглах
    """

    def __init__(self, data_dir: str = 'data/raw', checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the prediction system

        Args:
            data_dir: Directory containing CSV files
            checkpoint_dir: Directory for saving checkpoints
        """
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.sale_data: Optional[pd.DataFrame] = None
        self.rental_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

        # Models
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []

        # Results
        self.results: Dict[str, Dict[str, float]] = {}

        logger.info("PropertyPricePrediction initialized")

    def save_checkpoint(self, step: str, data: Any, filename: str = None) -> None:
        """
        Save checkpoint to resume from interruptions

        Args:
            step: Name of the processing step
            data: Data to save
            filename: Optional custom filename
        """
        try:
            if filename is None:
                filename = f"{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

            filepath = self.checkpoint_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filename: str) -> Any:
        """
        Load checkpoint to resume execution

        Args:
            filename: Checkpoint filename

        Returns:
            Loaded data
        """
        try:
            filepath = self.checkpoint_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Checkpoint loaded: {filepath}")
                return data
            else:
                logger.warning(f"Checkpoint not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load all CSV files from data directory

        Бүх property_cleaned_[date].csv файлуудыг ачаалж нэгтгэнэ

        Args:
            force_reload: Force reload even if checkpoint exists

        Returns:
            Combined DataFrame
        """
        checkpoint_file = 'raw_data_loaded.pkl'

        # Try to load from checkpoint
        if not force_reload:
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            if checkpoint_data is not None:
                self.raw_data = checkpoint_data
                logger.info(f"Loaded {len(self.raw_data)} records from checkpoint")
                return self.raw_data

        logger.info("Loading CSV files from data directory...")

        # Find all CSV files matching pattern
        csv_files = list(self.data_dir.glob('property_cleaned_*.csv'))

        if not csv_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            raise FileNotFoundError(f"No property_cleaned_*.csv files in {self.data_dir}")

        logger.info(f"Found {len(csv_files)} CSV files")

        # Load each file with progress bar
        dfs = []
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                dfs.append(df)
                logger.info(f"Loaded {csv_file.name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")

        # Combine all dataframes
        self.raw_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total records loaded: {len(self.raw_data)}")

        # Save checkpoint
        self.save_checkpoint('raw_data_loaded', self.raw_data, checkpoint_file)

        return self.raw_data

    def explore_data(self) -> None:
        """
        Perform initial data exploration

        Датаны анхны шинжилгээ хийх:
        - Хэмжээ, багана, төрөл
        - Алдагдсан утга
        - Үндсэн статистик үзүүлэлт
        """
        if self.raw_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return

        logger.info("\n" + "="*80)
        logger.info("DATA EXPLORATION")
        logger.info("="*80)

        # Basic info
        logger.info(f"\nDataset shape: {self.raw_data.shape}")
        logger.info(f"Columns: {list(self.raw_data.columns)}")

        # Data types
        logger.info("\nData Types:")
        logger.info(str(self.raw_data.dtypes))

        # Missing values
        logger.info("\nMissing Values:")
        missing = self.raw_data.isnull().sum()
        missing_pct = (missing / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        logger.info(str(missing_df))

        # Summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(str(self.raw_data.describe()))

        # Price distribution
        if 'price_in_mil' in self.raw_data.columns:
            logger.info("\nPrice Statistics:")
            logger.info(f"Mean price: {self.raw_data['price_in_mil'].mean():.2f} million")
            logger.info(f"Median price: {self.raw_data['price_in_mil'].median():.2f} million")
            logger.info(f"Min price: {self.raw_data['price_in_mil'].min():.2f} million")
            logger.info(f"Max price: {self.raw_data['price_in_mil'].max():.2f} million")

    def clean_data(self, force_clean: bool = False) -> pd.DataFrame:
        """
        Clean and preprocess the data

        Датаг цэвэрлэх:
        - Давхардсан зарыг устгах
        - Үнийн алдаа засах
        - Зарах/түрээслэх ялгах

        Args:
            force_clean: Force cleaning even if checkpoint exists

        Returns:
            Cleaned DataFrame
        """
        checkpoint_file = 'cleaned_data.pkl'

        # Try to load from checkpoint
        if not force_clean:
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            if checkpoint_data is not None:
                self.cleaned_data = checkpoint_data
                logger.info(f"Loaded cleaned data from checkpoint: {len(self.cleaned_data)} records")
                return self.cleaned_data

        if self.raw_data is None:
            logger.error("No raw data available. Call load_data() first.")
            return None

        logger.info("\n" + "="*80)
        logger.info("DATA CLEANING")
        logger.info("="*80)

        df = self.raw_data.copy()
        initial_count = len(df)

        # 1. Remove duplicates based on ID
        logger.info("\n1. Removing duplicate ads...")
        df = df.drop_duplicates(subset=['id'], keep='last')
        duplicates_removed = initial_count - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate ads")
        logger.info(f"Remaining records: {len(df)}")

        # 2. Convert ad_date to datetime
        logger.info("\n2. Converting date columns...")
        if 'ad_date' in df.columns:
            df['ad_date'] = pd.to_datetime(df['ad_date'], errors='coerce')

        # 3. Separate sale vs rental prices
        logger.info("\n3. Identifying sale vs rental listings...")

        # Rental prices are typically much lower (< 5 million MNT/month)
        # Sale prices are typically higher
        # We'll use distribution analysis to separate them

        if 'price_in_mil' in df.columns:
            # Remove extreme outliers first
            Q1 = df['price_in_mil'].quantile(0.01)
            Q3 = df['price_in_mil'].quantile(0.99)
            df = df[(df['price_in_mil'] >= Q1) & (df['price_in_mil'] <= Q3)]

            # Heuristic: rental prices < 5 million, sale prices >= 5 million
            # This can be adjusted based on domain knowledge
            rental_threshold = 5.0

            df['listing_type'] = df['price_in_mil'].apply(
                lambda x: 'rental' if x < rental_threshold else 'sale'
            )

            sale_count = (df['listing_type'] == 'sale').sum()
            rental_count = (df['listing_type'] == 'rental').sum()

            logger.info(f"Sale listings: {sale_count}")
            logger.info(f"Rental listings: {rental_count}")

        # 4. Handle missing values in key columns
        logger.info("\n4. Handling missing values...")

        # Fill numeric columns with median
        numeric_cols = ['sq_m', 'which_floor', 'total_floor', 'window_count',
                       'balcony', 'date_of_commission']
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                missing_before = df[col].isnull().sum()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"{col}: filled {missing_before} missing values with median ({median_val})")

        # Fill categorical columns with mode or 'Unknown'
        categorical_cols = ['location', 'district', 'neighbourhood', 'window_material',
                           'door_material', 'garage', 'elevator', 'payment_term']
        for col in categorical_cols:
            if col in df.columns:
                missing_before = df[col].isnull().sum()
                df[col].fillna('Unknown', inplace=True)
                logger.info(f"{col}: filled {missing_before} missing values with 'Unknown'")

        # 5. Create additional useful features
        logger.info("\n5. Creating derived features...")

        # Price per square meter
        if 'price_in_mil' in df.columns and 'sq_m' in df.columns:
            df['price_per_sqm'] = df['price_in_mil'] / df['sq_m']
            df['price_per_sqm'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Property age
        if 'date_of_commission' in df.columns:
            current_year = datetime.now().year
            df['property_age'] = current_year - df['date_of_commission']
            df['property_age'] = df['property_age'].clip(lower=0, upper=100)

        # Floor ratio (which floor / total floors)
        if 'which_floor' in df.columns and 'total_floor' in df.columns:
            df['floor_ratio'] = df['which_floor'] / df['total_floor']
            df['floor_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # 6. Remove rows with critical missing values
        logger.info("\n6. Removing rows with critical missing values...")
        critical_cols = ['price_in_mil', 'sq_m']
        before_drop = len(df)
        df = df.dropna(subset=critical_cols)
        after_drop = len(df)
        logger.info(f"Removed {before_drop - after_drop} rows with missing critical values")

        self.cleaned_data = df
        logger.info(f"\nCleaning complete. Final dataset: {len(self.cleaned_data)} records")

        # Save checkpoint
        self.save_checkpoint('cleaned_data', self.cleaned_data, checkpoint_file)

        # Save to CSV for inspection
        output_path = Path('data/processed/cleaned_data.csv')
        self.cleaned_data.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path}")

        return self.cleaned_data

    def perform_eda(self) -> None:
        """
        Perform Exploratory Data Analysis with visualizations

        Өгөгдлийн шинжилгээ болон дүрслэл:
        - Үнийн хуваарилалт
        - Байршлын шинжилгээ
        - Хамаарлын шинжилгээ
        """
        if self.cleaned_data is None:
            logger.error("No cleaned data available. Call clean_data() first.")
            return

        logger.info("\n" + "="*80)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("="*80)

        df = self.cleaned_data
        viz_dir = Path('visualizations')
        viz_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

        # 1. Price distribution
        logger.info("\n1. Analyzing price distribution...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        df['price_in_mil'].hist(bins=50, ax=axes[0], edgecolor='black')
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Price (Million MNT)')
        axes[0].set_ylabel('Frequency')

        # Box plot by listing type
        if 'listing_type' in df.columns:
            df.boxplot(column='price_in_mil', by='listing_type', ax=axes[1])
            axes[1].set_title('Price by Listing Type', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Listing Type')
            axes[1].set_ylabel('Price (Million MNT)')

        plt.tight_layout()
        plt.savefig(viz_dir / '01_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: 01_price_distribution.png")

        # 2. Price by location
        logger.info("\n2. Analyzing price by location...")
        if 'district' in df.columns:
            plt.figure(figsize=(14, 6))
            district_prices = df.groupby('district')['price_in_mil'].agg(['mean', 'median', 'count'])
            district_prices = district_prices.sort_values('median', ascending=False).head(15)

            district_prices['median'].plot(kind='barh', color='steelblue')
            plt.title('Median Price by District (Top 15)', fontsize=14, fontweight='bold')
            plt.xlabel('Median Price (Million MNT)')
            plt.ylabel('District')
            plt.tight_layout()
            plt.savefig(viz_dir / '02_price_by_district.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: 02_price_by_district.png")

        # 3. Correlation heatmap
        logger.info("\n3. Creating correlation heatmap...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / '03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: 03_correlation_heatmap.png")

        # 4. Price vs Square Meters
        logger.info("\n4. Analyzing price vs square meters...")
        plt.figure(figsize=(12, 6))
        plt.scatter(df['sq_m'], df['price_in_mil'], alpha=0.5, s=10)
        plt.title('Price vs Square Meters', fontsize=14, fontweight='bold')
        plt.xlabel('Square Meters')
        plt.ylabel('Price (Million MNT)')
        plt.tight_layout()
        plt.savefig(viz_dir / '04_price_vs_sqm.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: 04_price_vs_sqm.png")

        # 5. Ads posted over time
        logger.info("\n5. Analyzing ad posting trends...")
        if 'ad_date' in df.columns:
            df['ad_month'] = df['ad_date'].dt.to_period('M')
            ads_per_month = df.groupby('ad_month').size()

            plt.figure(figsize=(14, 6))
            ads_per_month.plot(kind='line', marker='o', linewidth=2, markersize=6)
            plt.title('Number of Ads Posted Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel('Number of Ads')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / '05_ads_over_time.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: 05_ads_over_time.png")

        # 6. Property age distribution
        logger.info("\n6. Analyzing property age distribution...")
        if 'property_age' in df.columns:
            plt.figure(figsize=(12, 6))
            df['property_age'].hist(bins=30, edgecolor='black', color='coral')
            plt.title('Property Age Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Property Age (Years)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(viz_dir / '06_property_age.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: 06_property_age.png")

        logger.info("\nEDA complete. All visualizations saved to: visualizations/")

    def prepare_features(self, target_type: str = 'sale') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training

        Модельд сургахад шаардлагатай features бэлдэх

        Args:
            target_type: 'sale' or 'rental'

        Returns:
            X (features), y (target)
        """
        if self.cleaned_data is None:
            logger.error("No cleaned data available.")
            return None, None

        logger.info(f"\nPreparing features for {target_type} price prediction...")

        # Filter by listing type
        if 'listing_type' in self.cleaned_data.columns:
            df = self.cleaned_data[self.cleaned_data['listing_type'] == target_type].copy()
        else:
            df = self.cleaned_data.copy()

        logger.info(f"Dataset size: {len(df)} records")

        # Select features
        feature_cols = []

        # Numeric features
        numeric_features = ['sq_m', 'which_floor', 'total_floor', 'window_count',
                          'balcony', 'date_of_commission', 'property_age',
                          'floor_ratio', 'price_per_sqm']

        for col in numeric_features:
            if col in df.columns and col != 'price_per_sqm':  # Don't use price_per_sqm as it's derived from target
                feature_cols.append(col)

        # Categorical features to encode
        categorical_features = ['district', 'neighbourhood', 'window_material',
                               'door_material', 'elevator', 'garage']

        # One-hot encode categorical variables
        for col in categorical_features:
            if col in df.columns:
                # Limit categories to top N to avoid too many features
                top_categories = df[col].value_counts().head(10).index
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')

                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())

        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['price_in_mil'].copy()

        # Handle any remaining missing values
        X.fillna(X.median(), inplace=True)

        self.feature_columns = feature_cols
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"Feature columns: {feature_cols[:10]}... (showing first 10)")

        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Train multiple regression models and compare performance

        Олон төрлийн модель сургаж харьцуулах

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set

        Returns:
            Dictionary of model results
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL TRAINING")
        logger.info("="*80)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logger.info(f"\nTraining set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Scale features
        logger.info("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15,
                                                   random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                           learning_rate=0.1, random_state=42)
        }

        results = {}

        # Train each model with progress bar
        for name, model in tqdm(models_to_train.items(), desc="Training models"):
            try:
                logger.info(f"\nTraining {name}...")

                # Train
                model.fit(X_train_scaled, y_train)

                # Predict
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)

                # Evaluate
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

                results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_mape': test_mape
                }

                # Save model
                self.models[name] = model

                logger.info(f"{name} Results:")
                logger.info(f"  Train R²: {train_r2:.4f}")
                logger.info(f"  Test R²: {test_r2:.4f}")
                logger.info(f"  Test RMSE: {test_rmse:.4f}")
                logger.info(f"  Test MAE: {test_mae:.4f}")
                logger.info(f"  Test MAPE: {test_mape:.2f}%")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        self.results = results

        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = self.models[best_model_name]

        logger.info("\n" + "="*80)
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
        logger.info(f"Test MAPE: {results[best_model_name]['test_mape']:.2f}%")
        logger.info("="*80)

        # Save results
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('test_r2', ascending=False)
        results_path = Path('models/model_comparison.csv')
        results_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(results_path)
        logger.info(f"\nModel comparison saved to: {results_path}")

        # Save best model
        model_path = Path('models/best_model.pkl')
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_name': best_model_name
        }, model_path)
        logger.info(f"Best model saved to: {model_path}")

        return results

    def predict_price(self, property_data: Dict[str, Any]) -> float:
        """
        Predict price for a new property

        Шинэ үл хөдлөх хөрөнгийн үнийг таамаглах

        Args:
            property_data: Dictionary with property features

        Returns:
            Predicted price in millions
        """
        if self.best_model is None or self.scaler is None:
            logger.error("No trained model available. Train a model first.")
            return None

        # Create feature vector
        # This would need to be implemented based on feature engineering
        logger.info("Predicting price for new property...")

        # For now, return placeholder
        # In production, you would transform property_data to match training features
        return None

    def run_complete_pipeline(self, target_type: str = 'sale') -> None:
        """
        Run the complete end-to-end pipeline

        Бүх процессыг эхнээс нь дуустал ажиллуулах

        Args:
            target_type: 'sale' or 'rental'
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("="*80)

        try:
            # Step 1: Load data
            self.load_data()

            # Step 2: Explore data
            self.explore_data()

            # Step 3: Clean data
            self.clean_data()

            # Step 4: Perform EDA
            self.perform_eda()

            # Step 5: Prepare features
            X, y = self.prepare_features(target_type=target_type)

            # Step 6: Train models
            self.train_models(X, y)

            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("Property Price Prediction System")
    print("Үл хөдлөх хөрөнгийн үнийг таамаглах систем")
    print("="*80 + "\n")

    # Initialize system
    predictor = PropertyPricePrediction(
        data_dir='data/raw',
        checkpoint_dir='checkpoints'
    )

    # Run complete pipeline
    # You can change target_type to 'rental' for rental price prediction
    predictor.run_complete_pipeline(target_type='sale')

    print("\n" + "="*80)
    print("All tasks completed!")
    print("Check the following directories:")
    print("  - visualizations/ : EDA plots and charts")
    print("  - models/ : Trained models and results")
    print("  - data/processed/ : Cleaned datasets")
    print("  - logs/ : Execution logs")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
