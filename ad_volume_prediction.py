"""
Ad Volume Prediction
====================
Зарын тоо хэрхэн өсөх буурах таамаглал

Time series forecasting for daily ad posting volume

Uses:
- Statistical methods (ARIMA)
- Facebook Prophet
- Simple trend analysis

Author: Claude AI
Date: 2025-11-18
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Time series libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdVolumePredictor:
    """
    Predict daily ad posting volume using time series analysis

    Өдөр бүр тавигдах зарын тоог таамаглах
    """

    def __init__(self, data_path: str = 'data/processed/cleaned_data.csv'):
        """
        Initialize predictor

        Args:
            data_path: Path to cleaned data CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        self.daily_counts = None
        self.prophet_model = None
        self.arima_model = None

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load data and aggregate by date

        Returns:
            DataFrame with daily ad counts
        """
        logger.info("Loading data...")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records")

        # Convert date column
        if 'ad_date' in self.df.columns:
            self.df['ad_date'] = pd.to_datetime(self.df['ad_date'], errors='coerce')

            # Aggregate by day
            daily_counts = self.df.groupby(self.df['ad_date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])

            # Sort by date
            daily_counts = daily_counts.sort_values('date').reset_index(drop=True)

            self.daily_counts = daily_counts
            logger.info(f"Aggregated to {len(daily_counts)} days")

            return daily_counts
        else:
            logger.error("No ad_date column found in data")
            return None

    def analyze_trends(self) -> None:
        """
        Analyze time series trends and patterns

        Цаг хугацааны чиг хандлага шинжлэх
        """
        if self.daily_counts is None:
            logger.error("No data loaded. Call load_and_prepare_data() first.")
            return

        logger.info("\n" + "="*80)
        logger.info("TIME SERIES ANALYSIS")
        logger.info("="*80)

        df = self.daily_counts

        # Basic statistics
        logger.info(f"\nDaily ad count statistics:")
        logger.info(f"Mean: {df['count'].mean():.2f}")
        logger.info(f"Median: {df['count'].median():.2f}")
        logger.info(f"Std: {df['count'].std():.2f}")
        logger.info(f"Min: {df['count'].min()}")
        logger.info(f"Max: {df['count'].max()}")

        # Calculate trend
        df['day_num'] = (df['date'] - df['date'].min()).dt.days

        # Simple linear trend
        z = np.polyfit(df['day_num'], df['count'], 1)
        trend_slope = z[0]

        if trend_slope > 0:
            logger.info(f"\nTrend: INCREASING ({trend_slope:.2f} ads/day)")
        elif trend_slope < 0:
            logger.info(f"\nTrend: DECREASING ({trend_slope:.2f} ads/day)")
        else:
            logger.info(f"\nTrend: STABLE")

        # Visualizations
        viz_dir = Path('visualizations')
        viz_dir.mkdir(exist_ok=True)

        # Plot 1: Daily ad volume over time
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Time series plot
        axes[0].plot(df['date'], df['count'], linewidth=2, color='steelblue')
        axes[0].set_title('Daily Ad Posting Volume', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Number of Ads')
        axes[0].grid(True, alpha=0.3)

        # Add trend line
        trend_line = z[0] * df['day_num'] + z[1]
        axes[0].plot(df['date'], trend_line, '--', color='red', linewidth=2, label='Trend')
        axes[0].legend()

        # Distribution
        axes[1].hist(df['count'], bins=30, edgecolor='black', color='coral')
        axes[1].set_title('Distribution of Daily Ad Counts', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Ads')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(viz_dir / '07_ad_volume_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: 07_ad_volume_analysis.png")

        # Decomposition (if enough data)
        if len(df) >= 30 and STATSMODELS_AVAILABLE:
            try:
                logger.info("\nPerforming seasonal decomposition...")
                df_ts = df.set_index('date')['count']

                # Use period of 7 days (weekly seasonality)
                decomposition = seasonal_decompose(df_ts, model='additive', period=7, extrapolate_trend='freq')

                fig, axes = plt.subplots(4, 1, figsize=(14, 12))

                decomposition.observed.plot(ax=axes[0], title='Observed')
                decomposition.trend.plot(ax=axes[1], title='Trend')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                decomposition.resid.plot(ax=axes[3], title='Residual')

                plt.tight_layout()
                plt.savefig(viz_dir / '08_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved: 08_seasonal_decomposition.png")
            except Exception as e:
                logger.warning(f"Could not perform decomposition: {e}")

    def train_prophet(self, forecast_days: int = 30) -> pd.DataFrame:
        """
        Train Facebook Prophet model for forecasting

        Args:
            forecast_days: Number of days to forecast

        Returns:
            Forecast DataFrame
        """
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping...")
            return None

        if self.daily_counts is None:
            logger.error("No data loaded.")
            return None

        logger.info("\n" + "="*80)
        logger.info("TRAINING PROPHET MODEL")
        logger.info("="*80)

        # Prepare data for Prophet
        prophet_df = self.daily_counts.copy()
        prophet_df.columns = ['ds', 'y']

        # Train model
        logger.info("\nTraining Prophet...")
        self.prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )

        self.prophet_model.fit(prophet_df)

        # Make future dataframe
        future = self.prophet_model.make_future_dataframe(periods=forecast_days)

        # Predict
        logger.info(f"Forecasting {forecast_days} days ahead...")
        forecast = self.prophet_model.predict(future)

        # Plot forecast
        fig = self.prophet_model.plot(forecast, figsize=(14, 6))
        plt.title('Ad Volume Forecast (Prophet)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Number of Ads')
        plt.tight_layout()
        plt.savefig('visualizations/09_prophet_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: 09_prophet_forecast.png")

        # Plot components
        fig = self.prophet_model.plot_components(forecast, figsize=(14, 8))
        plt.tight_layout()
        plt.savefig('visualizations/10_prophet_components.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: 10_prophet_components.png")

        # Extract future predictions
        future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        logger.info(f"\nForecast for next {forecast_days} days:")
        logger.info(f"Average predicted ads/day: {future_forecast['yhat'].mean():.2f}")
        logger.info(f"Range: {future_forecast['yhat'].min():.2f} - {future_forecast['yhat'].max():.2f}")

        # Save forecast
        forecast_path = Path('models/prophet_forecast.csv')
        forecast.to_csv(forecast_path, index=False)
        logger.info(f"\nFull forecast saved to: {forecast_path}")

        return forecast

    def train_arima(self, order: Tuple[int, int, int] = (1, 1, 1), forecast_days: int = 30) -> pd.DataFrame:
        """
        Train ARIMA model for forecasting

        Args:
            order: ARIMA (p, d, q) order
            forecast_days: Number of days to forecast

        Returns:
            Forecast array
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available, skipping...")
            return None

        if self.daily_counts is None:
            logger.error("No data loaded.")
            return None

        logger.info("\n" + "="*80)
        logger.info("TRAINING ARIMA MODEL")
        logger.info("="*80)

        # Prepare time series
        ts = self.daily_counts.set_index('date')['count']

        try:
            # Train ARIMA
            logger.info(f"\nTraining ARIMA{order}...")
            self.arima_model = ARIMA(ts, order=order)
            fitted_model = self.arima_model.fit()

            logger.info("\nModel Summary:")
            logger.info(str(fitted_model.summary()))

            # Forecast
            logger.info(f"\nForecasting {forecast_days} days ahead...")
            forecast = fitted_model.forecast(steps=forecast_days)

            # Create forecast dates
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

            # Plot
            plt.figure(figsize=(14, 6))
            plt.plot(ts.index, ts.values, label='Historical', linewidth=2)
            plt.plot(forecast_dates, forecast, label='Forecast', linewidth=2, color='red')
            plt.fill_between(forecast_dates, forecast * 0.9, forecast * 1.1, alpha=0.3, color='red')
            plt.title(f'Ad Volume Forecast (ARIMA{order})', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Number of Ads')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/11_arima_forecast.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: 11_arima_forecast.png")

            logger.info(f"\nForecast summary:")
            logger.info(f"Average predicted ads/day: {forecast.mean():.2f}")
            logger.info(f"Range: {forecast.min():.2f} - {forecast.max():.2f}")

            # Save forecast
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast
            })
            forecast_path = Path('models/arima_forecast.csv')
            forecast_df.to_csv(forecast_path, index=False)
            logger.info(f"\nForecast saved to: {forecast_path}")

            return forecast_df

        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return None

    def run_complete_analysis(self, forecast_days: int = 30) -> None:
        """
        Run complete ad volume analysis and forecasting

        Args:
            forecast_days: Days to forecast ahead
        """
        logger.info("\n" + "="*80)
        logger.info("AD VOLUME PREDICTION PIPELINE")
        logger.info("="*80)

        # Load data
        self.load_and_prepare_data()

        # Analyze trends
        self.analyze_trends()

        # Train Prophet
        if PROPHET_AVAILABLE:
            self.train_prophet(forecast_days=forecast_days)

        # Train ARIMA
        if STATSMODELS_AVAILABLE:
            self.train_arima(forecast_days=forecast_days)

        logger.info("\n" + "="*80)
        logger.info("AD VOLUME ANALYSIS COMPLETE")
        logger.info("="*80)


def main():
    """
    Main execution
    """
    print("\n" + "="*80)
    print("Ad Volume Prediction")
    print("Зарын тоо таамаглах")
    print("="*80 + "\n")

    # Check if data exists
    data_path = Path('data/processed/cleaned_data.csv')

    if not data_path.exists():
        print("Error: Cleaned data not found!")
        print("Please run property_price_prediction.py first.")
        return

    # Run prediction
    predictor = AdVolumePredictor(data_path)
    predictor.run_complete_analysis(forecast_days=30)

    print("\n" + "="*80)
    print("Ad volume prediction complete!")
    print("Check visualizations/ for forecast plots")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
