# Data Placement Guide
# Дата байршуулах заавар

This guide explains how to prepare and place your data files for the property price prediction system.

## Quick Start

1. **Create the data directory structure** (already done):
   ```
   data/
   └── raw/          <- Place your CSV files here
   ```

2. **Place your CSV files** in `data/raw/`:
   ```
   data/raw/
   ├── property_cleaned_20241121.csv
   ├── property_cleaned_20241201.csv
   ├── property_cleaned_20241215.csv
   └── property_cleaned_20250110.csv
   ```

3. **Run the pipeline**:
   ```bash
   python property_price_prediction.py
   ```

## CSV File Requirements

### File Naming Convention

Files must follow this pattern:
```
property_cleaned_YYYYMMDD.csv
```

Examples:
- ✓ `property_cleaned_20241121.csv`
- ✓ `property_cleaned_20241209.csv`
- ✗ `property_20241121.csv` (missing "cleaned")
- ✗ `cleaned_property_20241121.csv` (wrong order)

### Required Columns

Your CSV files MUST contain these columns:

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| `id` | int64 | Unique ad identifier | 123456 |
| `ad_title` | string | Ad title | "2 өрөө байр Баянзүрх" |
| `price_in_mil` | float64 | Price in millions MNT | 250.5 |
| `currency` | object | Currency code | "MNT" |
| `ad_link` | object | URL to ad | "https://..." |
| `ad_date` | datetime | Date ad was posted | "2024-11-21" |
| `location` | string | City/Location | "Улаанбаатар" |
| `district` | string | District name | "Баянзүрх" |
| `neighbourhood` | string | Neighbourhood | "4-р хороо" |
| `floor` | string | Floor description | "5-р давхар" |
| `balcony` | int | Number of balconies | 1 |
| `date_of_commission` | int | Year built | 2010 |
| `garage` | string | Garage availability | "Yes" / "No" |
| `window_material` | string | Window type | "Plastic" |
| `total_floor` | int | Total floors in building | 12 |
| `door_material` | string | Door type | "Metal" |
| `sq_m` | float64 | Square meters | 65.5 |
| `which_floor` | int | Which floor | 5 |
| `payment_term` | string | Payment terms | "Cash" |
| `window_count` | int | Number of windows | 3 |
| `elevator` | string | Elevator availability | "Yes" / "No" |

### Example CSV Format

```csv
id,ad_title,price_in_mil,currency,ad_link,ad_date,location,district,neighbourhood,floor,balcony,date_of_commission,garage,window_material,total_floor,door_material,sq_m,which_floor,payment_term,window_count,elevator
123456,"2 өрөө байр",250.5,MNT,https://...,2024-11-21,Улаанбаатар,Баянзүрх,4-р хороо,5-р давхар,1,2010,No,Plastic,12,Metal,65.5,5,Cash,3,Yes
123457,"3 өрөө байр",380.0,MNT,https://...,2024-11-22,Улаанбаатар,Сүхбаатар,1-р хороо,8-р давхар,2,2015,Yes,Plastic,16,Metal,85.0,8,Installment,4,Yes
```

## Data Quality Checklist

Before running the pipeline, verify:

- [ ] All CSV files are in `data/raw/` directory
- [ ] Files follow naming convention: `property_cleaned_YYYYMMDD.csv`
- [ ] Each file has all required columns
- [ ] Date column (`ad_date`) is in valid format
- [ ] Numeric columns contain valid numbers
- [ ] No completely empty files
- [ ] Files are readable (not corrupted)

## Common Issues and Solutions

### Issue 1: "No CSV files found"

**Problem**: Script can't find your data files

**Solution**:
1. Check files are in `data/raw/` directory
2. Verify file names match pattern: `property_cleaned_*.csv`
3. Check you're running script from project root directory

```bash
# Verify files
ls data/raw/

# Should show:
# property_cleaned_20241121.csv
# property_cleaned_20241201.csv
# ...
```

### Issue 2: Missing columns

**Problem**: CSV files don't have required columns

**Solution**:
1. Open CSV file and check column names
2. Ensure all required columns are present
3. Check for spelling mistakes in column names

### Issue 3: Date parsing errors

**Problem**: Dates are not recognized

**Solution**:
1. Ensure `ad_date` column is in format: `YYYY-MM-DD`
2. Valid examples:
   - ✓ `2024-11-21`
   - ✓ `2024-01-15`
   - ✗ `11/21/2024`
   - ✗ `21-11-2024`

### Issue 4: Duplicate data

**Problem**: Same ads appear multiple times

**Solution**:
- Don't worry! The script automatically removes duplicates based on `id` column
- It keeps the most recent version of each ad

### Issue 5: Mixed sale and rental prices

**Problem**: Sale and rental prices are mixed together

**Solution**:
- The script automatically separates them:
  - Prices < 5 million → Rental
  - Prices ≥ 5 million → Sale
- You can adjust this threshold if needed

## Data Volume Recommendations

### Minimum Requirements

- **Minimum records**: 1,000 listings (for basic model)
- **Recommended**: 10,000+ listings (for high accuracy)
- **Optimal**: 50,000+ listings (for 98% accuracy target)

### Time Period

- **Minimum**: 3 months of data
- **Recommended**: 12 months of data
- **Optimal**: 12+ months with recent data

## After Placing Data

Once you've placed your CSV files:

1. **Verify data is detected**:
   ```bash
   python quick_start.py
   # Choose option 5 to check data
   ```

2. **Run the pipeline**:
   ```bash
   python property_price_prediction.py
   ```

3. **Check results**:
   - `data/processed/cleaned_data.csv` - Cleaned and combined data
   - `visualizations/` - Charts and plots
   - `models/` - Trained models
   - `logs/property_prediction.log` - Detailed logs

## Example: Adding New Data

Let's say you downloaded new data on January 10, 2025:

1. **Save the file** with correct name:
   ```
   property_cleaned_20250110.csv
   ```

2. **Place it** in data directory:
   ```bash
   mv ~/Downloads/property_cleaned_20250110.csv data/raw/
   ```

3. **Verify**:
   ```bash
   ls data/raw/
   # Should show all your CSV files including the new one
   ```

4. **Run pipeline**:
   ```bash
   python property_price_prediction.py
   ```

The script will automatically:
- Detect the new file
- Combine it with existing data
- Remove any duplicates
- Train updated models

## Data Privacy

**Important**: This is local processing only
- All data stays on your computer
- No data is sent to external servers
- Models are trained locally
- Keep your data files secure

## Need Help?

If you're still having issues:

1. Check the log file: `logs/property_prediction.log`
2. Verify CSV file format matches examples above
3. Try with a small sample file first
4. Make sure all dependencies are installed: `pip install -r requirements.txt`

## Summary

1. ✓ Place CSV files in `data/raw/`
2. ✓ Follow naming: `property_cleaned_YYYYMMDD.csv`
3. ✓ Ensure all required columns exist
4. ✓ Run: `python property_price_prediction.py`
5. ✓ Check results in `visualizations/` and `models/`

Good luck! Амжилт хүсье! 🎉
