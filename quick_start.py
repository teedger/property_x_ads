#!/usr/bin/env python3
"""
Quick Start Script
==================
Хурдан эхлүүлэх скрипт

This script provides a simple interface to run the property price prediction pipeline.

Usage:
    python quick_start.py

Author: Claude AI
Date: 2025-11-18
"""

import os
import sys
from pathlib import Path


def check_environment():
    """Check if environment is properly set up"""
    print("Checking environment...")

    issues = []

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        issues.append(f"Python 3.8+ required. You have {python_version.major}.{python_version.minor}")

    # Check if data directory exists
    data_dir = Path('data/raw')
    if not data_dir.exists():
        issues.append("Data directory 'data/raw' not found")
    else:
        # Check for CSV files
        csv_files = list(data_dir.glob('property_cleaned_*.csv'))
        if not csv_files:
            issues.append("No CSV files found in data/raw/")
        else:
            print(f"✓ Found {len(csv_files)} CSV files")

    # Check if required packages are installed
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'tqdm']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        issues.append(f"Missing packages: {', '.join(missing_packages)}")
        issues.append("Run: pip install -r requirements.txt")

    return issues


def print_header():
    """Print welcome header"""
    print("\n" + "="*80)
    print("Property Price Prediction - Quick Start")
    print("Үл хөдлөх хөрөнгийн үнийг таамаглах - Хурдан эхлүүлэх")
    print("="*80 + "\n")


def print_menu():
    """Print main menu"""
    print("\nSelect an option:")
    print("1. Run complete pipeline (recommended)")
    print("2. Run basic models only")
    print("3. Run advanced models only")
    print("4. Run ad volume prediction only")
    print("5. Check data and visualizations")
    print("6. Exit")
    print("\nХарьцуулга:")
    print("1. Бүх процессыг ажиллуулах (санал болгох)")
    print("2. Зөвхөн үндсэн модель ажиллуулах")
    print("3. Зөвхөн өндөр түвшний модель ажиллуулах")
    print("4. Зөвхөн зарын тоо таамаглах")
    print("5. Дата болон дүрслэл шалгах")
    print("6. Гарах")


def run_complete_pipeline():
    """Run the complete pipeline"""
    print("\n" + "="*80)
    print("Running Complete Pipeline")
    print("="*80)

    try:
        from property_price_prediction import PropertyPricePrediction

        predictor = PropertyPricePrediction()
        predictor.run_complete_pipeline(target_type='sale')

        print("\n✓ Pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Check visualizations/ folder for plots")
        print("2. Check models/ folder for trained models")
        print("3. Check logs/ folder for detailed logs")
        print("\nTo run advanced models: python advanced_models.py")
        print("To predict ad volume: python ad_volume_prediction.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Check logs/property_prediction.log for details")


def run_basic_models():
    """Run basic models only"""
    print("\n" + "="*80)
    print("Running Basic Models")
    print("="*80)

    try:
        from property_price_prediction import PropertyPricePrediction

        predictor = PropertyPricePrediction()
        predictor.load_data()
        predictor.clean_data()
        X, y = predictor.prepare_features(target_type='sale')
        predictor.train_models(X, y)

        print("\n✓ Basic models completed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def run_advanced_models():
    """Run advanced models"""
    print("\n" + "="*80)
    print("Running Advanced Models")
    print("="*80)

    # Check if cleaned data exists
    if not Path('data/processed/cleaned_data.csv').exists():
        print("Error: Cleaned data not found!")
        print("Please run option 1 or 2 first to clean the data.")
        return

    try:
        import subprocess
        result = subprocess.run([sys.executable, 'advanced_models.py'],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✓ Advanced models completed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def run_ad_volume():
    """Run ad volume prediction"""
    print("\n" + "="*80)
    print("Running Ad Volume Prediction")
    print("="*80)

    # Check if cleaned data exists
    if not Path('data/processed/cleaned_data.csv').exists():
        print("Error: Cleaned data not found!")
        print("Please run option 1 or 2 first to clean the data.")
        return

    try:
        import subprocess
        result = subprocess.run([sys.executable, 'ad_volume_prediction.py'],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✓ Ad volume prediction completed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def check_results():
    """Check generated data and visualizations"""
    print("\n" + "="*80)
    print("Checking Results")
    print("="*80)

    # Check processed data
    processed_data = Path('data/processed/cleaned_data.csv')
    if processed_data.exists():
        import pandas as pd
        df = pd.read_csv(processed_data)
        print(f"\n✓ Cleaned data: {len(df)} records")
        print(f"  Location: {processed_data}")
    else:
        print("\n✗ No cleaned data found")
        print("  Run the pipeline first (option 1)")

    # Check models
    model_dir = Path('models')
    if model_dir.exists():
        models = list(model_dir.glob('*.pkl'))
        csvs = list(model_dir.glob('*.csv'))
        print(f"\n✓ Models: {len(models)} files")
        print(f"✓ Results: {len(csvs)} CSV files")
        for model in models:
            print(f"  - {model.name}")
    else:
        print("\n✗ No models found")

    # Check visualizations
    viz_dir = Path('visualizations')
    if viz_dir.exists():
        plots = list(viz_dir.glob('*.png'))
        print(f"\n✓ Visualizations: {len(plots)} plots")
        for plot in plots[:5]:  # Show first 5
            print(f"  - {plot.name}")
        if len(plots) > 5:
            print(f"  ... and {len(plots) - 5} more")
    else:
        print("\n✗ No visualizations found")

    # Check logs
    log_file = Path('logs/property_prediction.log')
    if log_file.exists():
        size = log_file.stat().st_size / 1024  # KB
        print(f"\n✓ Logs: {size:.2f} KB")
        print(f"  Location: {log_file}")
    else:
        print("\n✗ No logs found")


def main():
    """Main execution"""
    print_header()

    # Check environment
    issues = check_environment()
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before continuing.")

        if "CSV files" in str(issues):
            print("\n" + "="*80)
            print("How to add data files:")
            print("="*80)
            print("1. Place your CSV files in the data/raw/ directory")
            print("2. Files should be named: property_cleaned_YYYYMMDD.csv")
            print("3. Example: property_cleaned_20241121.csv")
            print("\nData directory structure:")
            print("data/")
            print("└── raw/")
            print("    ├── property_cleaned_20241121.csv")
            print("    ├── property_cleaned_20241201.csv")
            print("    └── ...")

        return

    print("✓ Environment check passed")

    # Main loop
    while True:
        print_menu()

        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == '1':
                run_complete_pipeline()
            elif choice == '2':
                run_basic_models()
            elif choice == '3':
                run_advanced_models()
            elif choice == '4':
                run_ad_volume()
            elif choice == '5':
                check_results()
            elif choice == '6':
                print("\nГөөгөө! Goodbye!")
                break
            else:
                print("\n✗ Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

        # Ask to continue
        if choice != '6':
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
