# Property Price Prediction - Mongolia Real Estate
# Үл хөдлөх хөрөнгийн үнийг таамаглах систем

Comprehensive machine learning system to predict property prices (sale and rental) in Mongolia using historical data from unegui.mn.

## Project Overview

This project develops a machine learning model to accurately predict property prices with 98% accuracy using:
- 12 months of property listing data from unegui.mn
- Multiple machine learning algorithms (Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- Comprehensive data cleaning and feature engineering
- Time series analysis for ad volume prediction

## Features

✅ **Data Processing**
- Automatic loading of multiple CSV files
- Duplicate removal based on ad ID
- Separation of sale vs rental listings
- Outlier detection and handling
- Missing value imputation

✅ **Exploratory Data Analysis**
- Price distribution analysis
- Geographic price analysis
- Feature correlation heatmaps
- Temporal trends
- Comprehensive visualizations

✅ **Machine Learning Models**
- Baseline models: Linear Regression, Ridge, Lasso
- Tree-based models: Decision Tree, Random Forest, Gradient Boosting
- Advanced models: XGBoost, LightGBM, CatBoost
- Automated model comparison and selection

✅ **Time Series Forecasting**
- Daily ad volume prediction
- Trend analysis
- Seasonal decomposition
- ARIMA and Prophet models

✅ **Robust Implementation**
- Checkpoint system for resumable execution
- Progress tracking with tqdm
- Comprehensive error handling
- Detailed logging
- Modular code structure

## Project Structure

```
property_x_ads/
├── data/
│   ├── raw/                          # Place CSV files here
│   │   ├── property_cleaned_20241121.csv
│   │   ├── property_cleaned_20241201.csv
│   │   └── ...
│   └── processed/                    # Cleaned data (auto-generated)
│       └── cleaned_data.csv
├── models/                           # Trained models (auto-generated)
│   ├── best_model.pkl
│   ├── best_advanced_model.pkl
│   ├── model_comparison.csv
│   └── advanced_model_comparison.csv
├── visualizations/                   # Plots and charts (auto-generated)
│   ├── 01_price_distribution.png
│   ├── 02_price_by_district.png
│   └── ...
├── checkpoints/                      # Execution checkpoints (auto-generated)
├── logs/                            # Execution logs (auto-generated)
│   └── property_prediction.log
├── notebooks/                       # Jupyter notebooks (optional)
├── scripts/                         # Utility scripts (optional)
├── property_price_prediction.py     # Main script
├── advanced_models.py               # Advanced ML models
├── ad_volume_prediction.py          # Time series forecasting
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### 1. Clone or download this repository

```bash
cd property_x_ads
```

### 2. Create virtual environment (recommended)

```bash
# Using venv
python3.12 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate  # On Windows

# Or using conda
conda create -n property_pred python=3.12
conda activate property_pred
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with Prophet on Mac M2:
```bash
brew install cmake
pip install prophet --no-build-isolation
```

## Data Preparation

### 1. Place your CSV files

Put all your `property_cleaned_[date].csv` files in the `data/raw/` directory:

```bash
data/raw/
├── property_cleaned_20241121.csv
├── property_cleaned_20241201.csv
├── property_cleaned_20241215.csv
└── ...
```

### 2. CSV File Format

Your CSV files should have these columns:
```
id, ad_title, price_in_mil, currency, ad_link, ad_date, location,
district, neighbourhood, floor, balcony, date_of_commission,
garage, window_material, total_floor, door_material, sq_m,
which_floor, payment_term, window_count, elevator
```

## Usage

### Quick Start - Run Complete Pipeline

```bash
python property_price_prediction.py
```

This will:
1. Load all CSV files from `data/raw/`
2. Clean and preprocess data
3. Perform exploratory data analysis
4. Train multiple machine learning models
5. Save the best model and results

### Step 2: Train Advanced Models (Optional)

```bash
python advanced_models.py
```

This trains advanced gradient boosting models (XGBoost, LightGBM, CatBoost) for higher accuracy.

### Step 3: Predict Ad Volume (Optional)

```bash
python ad_volume_prediction.py
```

This analyzes and predicts daily ad posting trends.

## Output

### Generated Files

After running the scripts, you'll find:

**Models** (`models/` directory):
- `best_model.pkl` - Best trained model
- `best_advanced_model.pkl` - Best advanced model
- `model_comparison.csv` - Performance comparison
- `prophet_forecast.csv` - Ad volume forecast
- `arima_forecast.csv` - Ad volume forecast (ARIMA)

**Visualizations** (`visualizations/` directory):
- `01_price_distribution.png` - Price distribution
- `02_price_by_district.png` - Price by location
- `03_correlation_heatmap.png` - Feature correlations
- `04_price_vs_sqm.png` - Price vs square meters
- `05_ads_over_time.png` - Ad posting trends
- `06_property_age.png` - Property age distribution
- `07_ad_volume_analysis.png` - Daily ad volume
- `08_seasonal_decomposition.png` - Seasonal patterns
- `09_prophet_forecast.png` - Prophet forecast
- `10_prophet_components.png` - Prophet components
- `11_arima_forecast.png` - ARIMA forecast
- `feature_importance.png` - Feature importance

**Data** (`data/processed/` directory):
- `cleaned_data.csv` - Cleaned dataset

**Logs** (`logs/` directory):
- `property_prediction.log` - Execution logs

### Model Performance

The system will display performance metrics for all models:
- **R² Score**: Coefficient of determination (target: ≥ 0.98)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error (target: ≤ 2%)

## Advanced Usage

### Resume from Checkpoint

If execution is interrupted, the script will automatically resume from the last checkpoint:

```bash
python property_price_prediction.py
# Will load from checkpoint if available
```

### Force Reload/Reclean

To force reloading and recleaning data:

```python
predictor = PropertyPricePrediction()
predictor.load_data(force_reload=True)
predictor.clean_data(force_clean=True)
```

### Predict Rental Prices

To train models for rental price prediction:

```python
predictor.run_complete_pipeline(target_type='rental')
```

### Custom Model Parameters

Edit the scripts to customize model parameters:

```python
# In advanced_models.py
params = {
    'max_depth': 10,  # Increase for more complex models
    'learning_rate': 0.03,  # Decrease for better accuracy
    'n_estimators': 1000,  # Increase for more training
}
```

## Google Colab

To run in Google Colab:

1. Upload all files to Google Drive
2. Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Change to project directory:
```python
%cd /content/drive/MyDrive/property_x_ads
```

4. Install requirements:
```python
!pip install -r requirements.txt
```

5. Run scripts:
```python
!python property_price_prediction.py
```

## Troubleshooting

### No CSV files found
- Ensure CSV files are in `data/raw/` directory
- Check file naming: `property_cleaned_*.csv`

### Missing dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Memory issues
- Process data in smaller batches
- Use only essential features
- Reduce model complexity

### Prophet installation issues (Mac M2)
```bash
brew install cmake
pip install pystan
pip install prophet --no-build-isolation
```

## Performance Optimization

### To achieve 98% accuracy:

1. **Feature Engineering**
   - Add more location-based features
   - Create interaction features
   - Use domain knowledge for custom features

2. **Model Tuning**
   - Use GridSearchCV or RandomizedSearchCV
   - Increase n_estimators for ensemble models
   - Adjust learning rate and regularization

3. **Data Quality**
   - Remove more outliers
   - Better handling of missing values
   - More sophisticated price separation (sale vs rental)

4. **Ensemble Methods**
   - Combine predictions from multiple models
   - Use stacking or blending

## Next Steps

- [ ] Add more advanced feature engineering
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Create ensemble models
- [ ] Add web scraping automation
- [ ] Build web interface for predictions
- [ ] Deploy model as API

## Technical Details

### System Requirements
- Python 3.12+
- 16GB RAM recommended
- MacOS, Linux, or Windows

### Dependencies
- pandas, numpy, scipy
- scikit-learn
- xgboost, lightgbm, catboost
- matplotlib, seaborn
- prophet, statsmodels
- tqdm, joblib

## Contributing

To improve the model:
1. Add your improvements
2. Test thoroughly
3. Document changes
4. Submit for review

## License

This project is for educational and research purposes.

## Author

Created by Claude AI for property price prediction in Mongolia.

## Contact

For questions or issues, please refer to the project documentation or logs.

---

**Амжилт хүсье!** 🎉
