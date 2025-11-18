#!/usr/bin/env python3
"""
Sample Data Generator
=====================
Туршилтын дата үүсгэгч

Generates realistic sample property data for testing the prediction system.
This is useful when you don't have real data yet.

Usage:
    python generate_sample_data.py

This will create sample CSV files in data/raw/ directory.

Author: Claude AI
Date: 2025-11-18
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_sample_data(num_records=5000, num_files=3):
    """
    Generate sample property data

    Args:
        num_records: Total number of property listings to generate
        num_files: Number of CSV files to create (split data across months)
    """
    print("="*80)
    print("Generating Sample Property Data")
    print("="*80)

    # Districts in Ulaanbaatar
    districts = [
        'Баянгол', 'Баянзүрх', 'Сонгинохайрхан', 'Сүхбаатар',
        'Хан-Уул', 'Чингэлтэй', 'Налайх', 'Багануур', 'Багахангай'
    ]

    # Neighbourhoods (generic)
    neighbourhoods = [f'{i}-р хороо' for i in range(1, 21)]

    # Window materials
    window_materials = ['Plastic', 'Wood', 'Aluminum', 'Unknown']

    # Door materials
    door_materials = ['Metal', 'Wood', 'MDF', 'Unknown']

    # Payment terms
    payment_terms = ['Cash', 'Installment', 'Mortgage', 'Mixed']

    # Generate base data
    records_per_file = num_records // num_files

    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data for each month
    start_date = datetime(2024, 11, 21)

    for file_idx in range(num_files):
        print(f"\nGenerating file {file_idx + 1}/{num_files}...")

        # Calculate date range for this file
        file_date = start_date + timedelta(days=30 * file_idx)
        date_str = file_date.strftime('%Y%m%d')

        data = []

        for i in range(records_per_file):
            # Decide if sale or rental (70% sale, 30% rental)
            is_sale = random.random() > 0.3

            # Property characteristics
            sq_m = np.random.normal(70, 25)  # Average 70 sqm, std 25
            sq_m = max(20, min(200, sq_m))  # Clip between 20-200

            total_floor = random.randint(5, 20)
            which_floor = random.randint(1, total_floor)

            year_built = random.randint(1990, 2024)
            property_age = 2024 - year_built

            # Number of rooms (affects price)
            num_rooms = random.choices([1, 2, 3, 4, 5], weights=[10, 40, 30, 15, 5])[0]

            # District (affects price significantly)
            district = random.choice(districts)

            # District price multipliers (some districts are more expensive)
            district_multipliers = {
                'Сүхбаатар': 1.3,
                'Хан-Уул': 1.25,
                'Чингэлтэй': 1.2,
                'Баянзүрх': 1.1,
                'Баянгол': 1.0,
                'Сонгинохайрхан': 0.95,
                'Налайх': 0.7,
                'Багануур': 0.65,
                'Багахангай': 0.6
            }

            district_mult = district_multipliers.get(district, 1.0)

            # Base price calculation
            if is_sale:
                # Sale price: ~3-4 million per room, affected by sqm, age, district
                base_price = num_rooms * 3.5  # million MNT
                sqm_factor = sq_m / 70  # Normalized to 70 sqm
                age_factor = max(0.7, 1 - (property_age / 100))  # Newer is better
                floor_factor = 1 + (which_floor / total_floor * 0.1)  # Higher floor slightly better

                price = base_price * sqm_factor * age_factor * district_mult * floor_factor
                price += np.random.normal(0, price * 0.1)  # Add some noise
                price = max(50, price)  # Minimum 50 million
            else:
                # Rental price: ~500k-1.5M per month depending on size and location
                base_rent = num_rooms * 0.5  # million MNT
                sqm_factor = sq_m / 70
                price = base_rent * sqm_factor * district_mult
                price += np.random.normal(0, price * 0.15)
                price = max(0.3, min(4.5, price))  # Clip between 0.3-4.5 million

            # Generate record
            ad_id = file_idx * records_per_file + i + 100000

            # Random date within the month
            days_offset = random.randint(0, 29)
            ad_date = file_date + timedelta(days=days_offset)

            record = {
                'id': ad_id,
                'ad_title': f'{num_rooms} өрөө байр {district}',
                'price_in_mil': round(price, 2),
                'currency': 'MNT',
                'ad_link': f'https://unegui.mn/ad/{ad_id}',
                'ad_date': ad_date.strftime('%Y-%m-%d'),
                'location': 'Улаанбаатар',
                'district': district,
                'neighbourhood': random.choice(neighbourhoods),
                'floor': f'{which_floor}-р давхар',
                'balcony': random.randint(0, 2),
                'date_of_commission': year_built,
                'garage': random.choice(['Yes', 'No', 'Unknown']),
                'window_material': random.choice(window_materials),
                'total_floor': total_floor,
                'door_material': random.choice(door_materials),
                'sq_m': round(sq_m, 1),
                'which_floor': which_floor,
                'payment_term': random.choice(payment_terms),
                'window_count': random.randint(2, 6),
                'elevator': 'Yes' if total_floor > 5 else random.choice(['Yes', 'No'])
            }

            data.append(record)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add some duplicate records (5%)
        num_duplicates = int(len(df) * 0.05)
        if num_duplicates > 0:
            duplicates = df.sample(n=num_duplicates).copy()
            # Slightly modify price for duplicates
            duplicates['price_in_mil'] *= np.random.uniform(0.98, 1.02, len(duplicates))
            df = pd.concat([df, duplicates], ignore_index=True)

        # Shuffle
        df = df.sample(frac=1).reset_index(drop=True)

        # Add some missing values (realistic data quality issues)
        missing_rate = 0.05
        for col in ['balcony', 'garage', 'window_material', 'door_material', 'elevator']:
            mask = np.random.random(len(df)) < missing_rate
            if col in ['balcony']:
                df.loc[mask, col] = np.nan
            else:
                df.loc[mask, col] = None

        # Save to CSV
        filename = f'property_cleaned_{date_str}.csv'
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)

        print(f"✓ Created: {filepath}")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['ad_date'].min()} to {df['ad_date'].max()}")
        print(f"  Price range: {df['price_in_mil'].min():.2f} - {df['price_in_mil'].max():.2f} million MNT")

    print("\n" + "="*80)
    print("Sample Data Generation Complete!")
    print("="*80)
    print(f"\nGenerated {num_files} CSV files with ~{num_records} total records")
    print(f"Location: {output_dir}/")
    print("\nYou can now run:")
    print("  python property_price_prediction.py")
    print("\nNote: This is synthetic data for testing purposes only.")


def main():
    """Main execution"""
    print("\nThis script generates synthetic property data for testing.\n")

    # Ask user for parameters
    try:
        num_records = int(input("Enter total number of records to generate (default: 5000): ") or "5000")
        num_files = int(input("Enter number of CSV files to create (default: 3): ") or "3")
    except ValueError:
        print("Invalid input. Using defaults.")
        num_records = 5000
        num_files = 3

    # Confirm
    print(f"\nWill generate {num_records} records across {num_files} files.")
    confirm = input("Continue? (y/n): ").lower()

    if confirm == 'y':
        generate_sample_data(num_records=num_records, num_files=num_files)
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
