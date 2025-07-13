# Complete Pandas Tutorial: From Scratch to Real-World Projects

## Prerequisites Check ✓
- Python programming: ✓
- Medium-level NumPy: ✓
- Dataset ready: Social Media Addiction from Kaggle ✓

---

## Module 1: Getting Started with Pandas

### 1.1 Installation and Import
```python
# Install pandas (if not already installed)
# pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

### 1.2 Loading Your Dataset
```python
# Load your social media addiction dataset
# Replace 'your_dataset.csv' with your actual filename
df = pd.read_csv('social_media_addiction.csv')

# Alternative loading methods you might need:
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')
# df = pd.read_csv('data.csv', encoding='utf-8', sep=';')  # for different separators/encodings
```

**Practice Exercise 1**: Load your dataset and try different parameters like `encoding='latin-1'` if you get encoding errors.

---

## Module 2: Data Inspection and Exploration

### 2.1 First Look at Your Data
```python
# Basic information about your dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# First and last few rows
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Random sample
print("\nRandom 5 rows:")
print(df.sample(5))
```

### 2.2 Dataset Overview
```python
# Comprehensive dataset information
print("Dataset Info:")
df.info()

print("\nMemory usage:")
print(df.memory_usage(deep=True))

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# For categorical columns
print("\nCategorical columns summary:")
print(df.describe(include='object'))
```

### 2.3 Data Types and Structure
```python
# Check data types
print("Data types:")
print(df.dtypes)

# Check for mixed types
for col in df.columns:
    unique_types = df[col].apply(type).unique()
    if len(unique_types) > 1:
        print(f"Column '{col}' has mixed types: {unique_types}")

# Check unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n'{col}' unique values ({df[col].nunique()}):")
    print(df[col].value_counts().head(10))
```

**Practice Exercise 2**: Run these commands on your dataset and identify:
- How many rows and columns you have
- What types of data (numerical, categorical, dates)
- Any obvious data quality issues

---

## Module 3: Data Cleaning and Quality Assessment

### 3.1 Missing Data Analysis
```python
# Check for missing values
print("Missing values count:")
missing_count = df.isnull().sum()
print(missing_count[missing_count > 0])

# Missing values percentage
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percent
})
print("\nMissing values summary:")
print(missing_summary[missing_summary['Missing Count'] > 0])

# Visualize missing data pattern
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
df.isnull().sum().plot(kind='bar')
plt.title('Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Missing Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3.2 Handling Missing Values
```python
# Strategy 1: Drop rows with missing values
df_dropped = df.dropna()
print(f"Original shape: {df.shape}, After dropping: {df_dropped.shape}")

# Strategy 2: Drop columns with too many missing values (>50%)
threshold = len(df) * 0.5
df_clean = df.dropna(axis=1, thresh=threshold)

# Strategy 3: Fill missing values
# For numerical columns - use mean, median, or mode
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().any():
        # Fill with median (more robust to outliers)
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns - use mode or 'Unknown'
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        # Fill with mode (most frequent value)
        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_value, inplace=True)

# Advanced: Forward fill and backward fill for time series
# df['column_name'].fillna(method='ffill', inplace=True)
# df['column_name'].fillna(method='bfill', inplace=True)
```

### 3.3 Duplicate Detection and Removal
```python
# Check for duplicates
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Find duplicate rows
duplicates = df[df.duplicated(keep=False)]
if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)

# Remove duplicates
df_no_duplicates = df.drop_duplicates()
print(f"Shape after removing duplicates: {df_no_duplicates.shape}")

# Check for duplicates based on specific columns
# df.duplicated(subset=['user_id', 'timestamp']).sum()
```

**Practice Exercise 3**: Clean your dataset by:
- Identifying missing values patterns
- Choosing appropriate filling strategies
- Removing duplicates if any

---

## Module 4: Data Type Conversions

### 4.1 Converting Data Types
```python
# Check current data types
print("Current data types:")
print(df.dtypes)

# Convert strings to numbers
# If you have columns that should be numeric but are stored as strings
# df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Convert to datetime
# If you have date columns
# df['date_column'] = pd.to_datetime(df['date_column'])

# Convert to categorical (saves memory for repeated strings)
# df['platform'] = df['platform'].astype('category')

# Example with social media data
# Assuming your dataset has these types of columns:

# Convert age to numeric if it's stored as string
if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Convert categorical columns to category type
categorical_columns = ['platform', 'gender', 'location']  # adjust based on your data
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Convert boolean columns
boolean_columns = ['is_addicted', 'has_notifications']  # adjust based on your data
for col in boolean_columns:
    if col in df.columns:
        df[col] = df[col].astype('bool')
```

### 4.2 String Operations
```python
# Common string operations for data cleaning
# Assuming you have text columns that need cleaning

if 'platform' in df.columns:
    # Clean platform names
    df['platform'] = df['platform'].str.strip()  # Remove whitespace
    df['platform'] = df['platform'].str.lower()  # Lowercase
    df['platform'] = df['platform'].str.title()  # Title case

# Extract information from strings
# df['domain'] = df['url'].str.extract(r'https?://([^/]+)')

# Replace values
# df['platform'] = df['platform'].str.replace('fb', 'facebook')
```

**Practice Exercise 4**: 
- Convert appropriate columns to correct data types
- Clean any text columns in your dataset
- Check memory usage before and after conversions

---

## Module 5: Indexing and Data Selection

### 5.1 Basic Selection
```python
# Select single column
platform_data = df['platform']  # Returns Series
platform_df = df[['platform']]  # Returns DataFrame

# Select multiple columns
selected_cols = df[['platform', 'age', 'daily_usage_hours']]

# Select rows by position
first_10_rows = df.iloc[0:10]
last_5_rows = df.iloc[-5:]

# Select specific rows and columns
specific_data = df.iloc[0:10, 1:4]  # First 10 rows, columns 1-3
```

### 5.2 Label-based Selection with .loc
```python
# Select by row labels (if you set custom index)
# df.loc['row_label']

# Select by column labels
age_data = df.loc[:, 'age']  # All rows, age column
subset = df.loc[:, 'platform':'age']  # All rows, from platform to age

# Select specific rows and columns
subset = df.loc[0:10, ['platform', 'age']]
```

### 5.3 Boolean Indexing (Most Important!)
```python
# Filter rows based on conditions
# Users over 25
adults = df[df['age'] > 25]

# Multiple conditions
heavy_users = df[(df['age'] > 18) & (df['daily_usage_hours'] > 5)]

# String conditions
instagram_users = df[df['platform'] == 'Instagram']
social_platforms = df[df['platform'].isin(['Facebook', 'Instagram', 'Twitter'])]

# Complex conditions
addicted_young = df[
    (df['age'] < 25) & 
    (df['daily_usage_hours'] > 6) & 
    (df['platform'].isin(['Instagram', 'TikTok']))
]

# Using query method (alternative syntax)
result = df.query('age > 25 and daily_usage_hours > 3')
```

### 5.4 Advanced Selection
```python
# Select based on string patterns
# Users from cities containing 'New'
# city_users = df[df['city'].str.contains('New', na=False)]

# Select rows with missing values
rows_with_missing = df[df.isnull().any(axis=1)]

# Select rows without missing values
complete_rows = df[df.notnull().all(axis=1)]
```

**Practice Exercise 5**: Create these subsets from your data:
- Users above average age
- Users with high social media usage
- Users from specific platforms
- Users meeting multiple criteria

---

## Module 6: Data Transformation and Feature Engineering

### 6.1 Creating New Columns
```python
# Simple calculations
df['usage_category'] = df['daily_usage_hours'].apply(
    lambda x: 'Low' if x < 2 else 'Medium' if x < 5 else 'High'
)

# Age groups
df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 18, 25, 35, 50, 100], 
                        labels=['<18', '18-25', '26-35', '36-50', '50+'])

# Binary features
df['is_heavy_user'] = (df['daily_usage_hours'] > df['daily_usage_hours'].median()).astype(int)

# Multiple platform usage
# df['platforms_count'] = df.groupby('user_id')['platform'].transform('nunique')
```

### 6.2 Apply Functions
```python
# Apply function to single column
def categorize_usage(hours):
    if hours < 1:
        return 'Minimal'
    elif hours < 3:
        return 'Light'
    elif hours < 6:
        return 'Moderate'
    else:
        return 'Heavy'

df['usage_level'] = df['daily_usage_hours'].apply(categorize_usage)

# Apply to multiple columns
def calculate_addiction_score(row):
    score = 0
    score += row['daily_usage_hours'] * 10
    score += row['notifications_per_day'] * 0.1
    if row['age'] < 25:
        score *= 1.2  # Young people more susceptible
    return score

df['addiction_score'] = df.apply(calculate_addiction_score, axis=1)
```

### 6.3 String Manipulations
```python
# Assuming you have text data to work with
if 'platform' in df.columns:
    # Extract first letter
    df['platform_initial'] = df['platform'].str[0]
    
    # Length of platform name
    df['platform_name_length'] = df['platform'].str.len()
    
    # Check if contains specific words
    df['is_meta_platform'] = df['platform'].str.contains('Facebook|Instagram', na=False)
```

**Practice Exercise 6**: Create these new features:
- Usage intensity categories
- Age groups
- Addiction risk score
- Platform type categories

---

## Module 7: Grouping and Aggregation

### 7.1 Basic Grouping
```python
# Group by single column
platform_stats = df.groupby('platform').agg({
    'daily_usage_hours': ['mean', 'median', 'std'],
    'age': ['mean', 'count'],
    'addiction_score': 'mean'
})

print("Platform Statistics:")
print(platform_stats)

# Group by multiple columns
age_platform_stats = df.groupby(['age_group', 'platform']).agg({
    'daily_usage_hours': 'mean',
    'addiction_score': 'mean'
}).round(2)

print("\nAge Group and Platform Statistics:")
print(age_platform_stats)
```

### 7.2 Advanced Aggregations
```python
# Custom aggregation functions
def usage_range(series):
    return series.max() - series.min()

platform_analysis = df.groupby('platform').agg({
    'daily_usage_hours': [
        'mean', 
        'median', 
        'std', 
        usage_range,
        lambda x: x.quantile(0.95)  # 95th percentile
    ],
    'age': ['min', 'max', 'mean'],
    'user_id': 'count'  # Count of users per platform
})

# Rename columns for clarity
platform_analysis.columns = [
    'avg_usage', 'median_usage', 'std_usage', 'usage_range', '95th_percentile',
    'min_age', 'max_age', 'avg_age', 'user_count'
]

print("Detailed Platform Analysis:")
print(platform_analysis)
```

### 7.3 Transform and Filter Groups
```python
# Transform: add group statistics back to original dataframe
df['platform_avg_usage'] = df.groupby('platform')['daily_usage_hours'].transform('mean')
df['usage_vs_platform_avg'] = df['daily_usage_hours'] - df['platform_avg_usage']

# Filter: keep only groups meeting certain criteria
# Keep only platforms with more than 100 users
popular_platforms = df.groupby('platform').filter(lambda x: len(x) > 100)

# Keep only age groups with high average usage
high_usage_ages = df.groupby('age_group').filter(
    lambda x: x['daily_usage_hours'].mean() > 4
)
```

**Practice Exercise 7**: Analyze your data by:
- Platform usage patterns
- Age group behaviors
- Gender differences (if available)
- Creating group-based features

---

## Module 8: Merging and Joining Data

### 8.1 Concatenating DataFrames
```python
# Split your data for practice
df1 = df.iloc[:len(df)//2]  # First half
df2 = df.iloc[len(df)//2:]  # Second half

# Concatenate vertically (stack rows)
combined = pd.concat([df1, df2], ignore_index=True)
print(f"Original: {df.shape}, Combined: {combined.shape}")

# Concatenate horizontally (side by side)
# Create a simple second dataframe for demo
df_extra = pd.DataFrame({
    'user_id': df['user_id'] if 'user_id' in df.columns else range(len(df)),
    'premium_user': np.random.choice([True, False], len(df))
})

# Horizontal concatenation
df_wide = pd.concat([df, df_extra[['premium_user']]], axis=1)
```

### 8.2 Merging DataFrames
```python
# Create sample related data for merging practice
platform_info = pd.DataFrame({
    'platform': ['Facebook', 'Instagram', 'Twitter', 'TikTok', 'Snapchat'],
    'founded_year': [2004, 2010, 2006, 2016, 2011],
    'parent_company': ['Meta', 'Meta', 'X Corp', 'ByteDance', 'Snap Inc'],
    'primary_demographic': ['All ages', 'Young adults', 'News/Politics', 'Gen Z', 'Teens']
})

# Inner join (only matching records)
merged_inner = df.merge(platform_info, on='platform', how='inner')

# Left join (keep all records from left dataframe)
merged_left = df.merge(platform_info, on='platform', how='left')

# Check results
print(f"Original df: {df.shape}")
print(f"Platform info: {platform_info.shape}")
print(f"Inner merge: {merged_inner.shape}")
print(f"Left merge: {merged_left.shape}")
```

### 8.3 Advanced Merging
```python
# Merge on multiple columns (if applicable)
# merged = df1.merge(df2, on=['user_id', 'date'], how='inner')

# Merge with different column names
# merged = df1.merge(df2, left_on='user_id', right_on='id', how='left')

# Merge with index
# merged = df1.merge(df2, left_index=True, right_index=True)
```

**Practice Exercise 8**: 
- Create additional datasets related to your main data
- Practice different types of joins
- Analyze the impact of different merge strategies

---

## Module 9: Time Series Operations (If applicable)

### 9.1 Working with Dates
```python
# If your dataset has date columns, convert them
# df['date'] = pd.to_datetime(df['date_column'])

# If you need to create sample dates for practice
df['sample_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Extract date components
df['year'] = df['sample_date'].dt.year
df['month'] = df['sample_date'].dt.month
df['day_of_week'] = df['sample_date'].dt.day_name()
df['is_weekend'] = df['sample_date'].dt.weekday.isin([5, 6])

# Set date as index
df_time = df.set_index('sample_date')
```

### 9.2 Time-based Grouping
```python
# Group by time periods
monthly_usage = df.groupby(df['sample_date'].dt.to_period('M'))['daily_usage_hours'].mean()
weekly_patterns = df.groupby(df['sample_date'].dt.day_name())['daily_usage_hours'].mean()

print("Average usage by day of week:")
print(weekly_patterns.sort_values(ascending=False))
```

**Practice Exercise 9**: If your data has timestamps:
- Extract time components
- Analyze usage patterns by time
- Create time-based visualizations

---

## Module 10: Performance Optimization

### 10.1 Memory Optimization
```python
# Check memory usage
print("Memory usage by column:")
print(df.memory_usage(deep=True))

# Optimize categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].nunique() < df.shape[0] * 0.5:  # Less than 50% unique values
        df[col] = df[col].astype('category')

# Optimize numerical columns
for col in df.select_dtypes(include=['int64']).columns:
    col_min = df[col].min()
    col_max = df[col].max()
    
    if col_min >= 0:  # Unsigned integers
        if col_max < 255:
            df[col] = df[col].astype('uint8')
        elif col_max < 65535:
            df[col] = df[col].astype('uint16')
        elif col_max < 4294967295:
            df[col] = df[col].astype('uint32')
    else:  # Signed integers
        if col_min > -128 and col_max < 127:
            df[col] = df[col].astype('int8')
        elif col_min > -32768 and col_max < 32767:
            df[col] = df[col].astype('int16')
        elif col_min > -2147483648 and col_max < 2147483647:
            df[col] = df[col].astype('int32')

print("Memory usage after optimization:")
print(df.memory_usage(deep=True))
```

### 10.2 Vectorized Operations
```python
# Instead of loops, use vectorized operations
# Bad: Using loops
# addiction_risk = []
# for index, row in df.iterrows():
#     if row['daily_usage_hours'] > 5 and row['age'] < 25:
#         addiction_risk.append('High')
#     else:
#         addiction_risk.append('Low')

# Good: Vectorized operation
df['addiction_risk'] = np.where(
    (df['daily_usage_hours'] > 5) & (df['age'] < 25),
    'High',
    'Low'
)

# Using .loc for conditional assignment
df.loc[(df['daily_usage_hours'] > 5) & (df['age'] < 25), 'addiction_risk'] = 'High'
df.loc[~((df['daily_usage_hours'] > 5) & (df['age'] < 25)), 'addiction_risk'] = 'Low'
```

---

## Module 11: Data Visualization Integration

### 11.1 Quick Plots with Pandas
```python
# Basic plotting
df['daily_usage_hours'].hist(bins=30, figsize=(10, 6))
plt.title('Distribution of Daily Usage Hours')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.show()

# Box plot by category
df.boxplot(column='daily_usage_hours', by='platform', figsize=(12, 8))
plt.title('Usage Hours by Platform')
plt.show()

# Correlation heatmap
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

### 11.2 Advanced Visualizations
```python
# Group statistics visualization
platform_stats = df.groupby('platform')['daily_usage_hours'].agg(['mean', 'std']).reset_index()
platform_stats.plot(x='platform', y='mean', kind='bar', figsize=(12, 6))
plt.title('Average Usage by Platform')
plt.ylabel('Hours')
plt.xticks(rotation=45)
plt.show()

# Scatter plot with categories
plt.figure(figsize=(12, 8))
for platform in df['platform'].unique():
    platform_data = df[df['platform'] == platform]
    plt.scatter(platform_data['age'], platform_data['daily_usage_hours'], 
                label=platform, alpha=0.7)

plt.xlabel('Age')
plt.ylabel('Daily Usage Hours')
plt.title('Age vs Usage by Platform')
plt.legend()
plt.show()
```

---

## Module 12: Real-World Project Workflow

### 12.1 Complete Analysis Pipeline
```python
def analyze_social_media_data(df):
    """
    Complete analysis pipeline for social media addiction data
    """
    print("=== SOCIAL MEDIA ADDICTION ANALYSIS ===\n")
    
    # 1. Data Overview
    print("1. DATA OVERVIEW")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    # 2. Data Quality Check
    print("2. DATA QUALITY")
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing values found:")
        print(missing_data[missing_data > 0])
    else:
        print("No missing values found")
    
    print(f"Duplicate rows: {df.duplicated().sum()}\n")
    
    # 3. Key Statistics
    print("3. KEY STATISTICS")
    if 'daily_usage_hours' in df.columns:
        print(f"Average daily usage: {df['daily_usage_hours'].mean():.2f} hours")
        print(f"Median daily usage: {df['daily_usage_hours'].median():.2f} hours")
        print(f"Max daily usage: {df['daily_usage_hours'].max():.2f} hours")
    
    if 'age' in df.columns:
        print(f"Average age: {df['age'].mean():.1f} years")
        print(f"Age range: {df['age'].min()}-{df['age'].max()} years")
    
    if 'platform' in df.columns:
        print(f"Most popular platform: {df['platform'].mode()[0]}")
        print(f"Number of platforms: {df['platform'].nunique()}\n")
    
    # 4. Platform Analysis
    if 'platform' in df.columns and 'daily_usage_hours' in df.columns:
        print("4. PLATFORM ANALYSIS")
        platform_stats = df.groupby('platform').agg({
            'daily_usage_hours': ['mean', 'count'],
            'age': 'mean'
        }).round(2)
        platform_stats.columns = ['avg_usage', 'user_count', 'avg_age']
        print(platform_stats.sort_values('avg_usage', ascending=False))
        print()
    
    # 5. Risk Assessment
    if 'daily_usage_hours' in df.columns:
        print("5. ADDICTION RISK ASSESSMENT")
        high_risk = df[df['daily_usage_hours'] > 6].shape[0]
        total_users = df.shape[0]
        risk_percentage = (high_risk / total_users) * 100
        print(f"High-risk users (>6 hours/day): {high_risk} ({risk_percentage:.1f}%)")
        
        if 'age' in df.columns:
            young_high_risk = df[(df['age'] < 25) & (df['daily_usage_hours'] > 6)].shape[0]
            young_total = df[df['age'] < 25].shape[0]
            if young_total > 0:
                young_risk_pct = (young_high_risk / young_total) * 100
                print(f"High-risk young users (<25): {young_high_risk} ({young_risk_pct:.1f}%)")
    
    return df

# Run the analysis
analyzed_df = analyze_social_media_data(df)
```

### 12.2 Export Results
```python
# Export cleaned data
df.to_csv('cleaned_social_media_data.csv', index=False)

# Export analysis results
platform_summary = df.groupby('platform').agg({
    'daily_usage_hours': ['mean', 'median', 'std', 'count'],
    'age': 'mean'
}).round(2)

platform_summary.to_csv('platform_analysis.csv')

print("Analysis complete! Files saved:")
print("- cleaned_social_media_data.csv")
print("- platform_analysis.csv")
```

---

## Next Steps for Machine Learning

### Preparing Data for ML
```python
# Feature encoding for machine learning
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
le = LabelEncoder()
df_ml = df.copy()

categorical_cols = df_ml.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))

# Scale numerical features
scaler = StandardScaler()
numerical_cols = df_ml.select_dtypes(include=[np.number]).columns
df_ml[numerical_cols] = scaler.fit_transform(df_ml[numerical_cols])

print("Data prepared for machine learning!")
print(f"Final shape: {df_ml.shape}")
```

---

## Summary and Best Practices

### Key Takeaways:
1. **Always start with data exploration** - understand your data before manipulating it
2. **Handle missing values thoughtfully** - choose strategies based on data context
3. **Use vectorized operations** - avoid loops for better performance
4. **Create meaningful features** - domain knowledge drives good feature engineering
5. **Validate your transformations** - check results after each major operation
6. **Document your process** - keep track of transformations for reproducibility

### Common Freelancing Scenarios:
- **Client data exploration**: Use inspection methods to quickly understand new datasets
- **Data cleaning pipelines**: Create reusable functions for common cleaning tasks
- **Feature engineering**: Domain-specific transformations that add business value
- **Performance optimization**: Essential for large datasets in production
- **Reporting and visualization**: Clear communication of findings to non-technical clients

### Practice Projects:
1. **Customer Segmentation**: Group customers by behavior patterns
2. **Time Series Analysis**: Analyze trends and seasonality
3. **A/B Testing**: Compare different groups statistically
4. **Churn Prediction**: Identify at-risk customers
5. **Market Basket Analysis**: Find product associations

Remember: The best way to learn pandas is by working with real data on projects that interest you. Keep practicing with your social media dataset and gradually take on more complex analyses!