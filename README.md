# -nelises-neural-network-time-series-forecasting-
Examining Time Series Forecasting Approaches for Predicting Canadian Employment Trends

# Wage Data File
## Due to the file size, it could not be uploaded directly to the repository, but it can be found here:
https://drive.google.com/file/d/1h-IQSK-YRBwkyasyJsfX56jIzjcepYHf/view?usp=drive_link

# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('Unemployment (Selected Variables).csv')

# Check the first five records
first_five_records = data.head()

print('First Five Records\n', first_five_records)

# Drop unnecessary columns
columns_to_drop = ['DGUID', 'UOM_ID', 'SCALAR_ID', 'COORDINATE', 'VECTOR', 'STATUS', 'SYMBOL', 'TERMINATED', 'DECIMALS']
data_emp = data.drop(columns=columns_to_drop)

# Separating the 'North American Industry Classification System (NAICS)' into two columns
data_emp[['industry_classification', 'NAICS']] = data_emp['North American Industry Classification System (NAICS)'].str.extract(r'([^\[]+)\s*\[([^\]]+)\]')
data_emp = data_emp.drop(columns=['North American Industry Classification System (NAICS)'])

# Ensure new columns are properly formatted
data_emp['industry_classification'] = data_emp['industry_classification'].str.strip().astype('category')
data_emp['NAICS'] = data_emp['NAICS'].str.strip()

# Standardizing Data Formats
categorical_columns = ['GEO', 'Labour force characteristics', 'Sex', 'Age group', 'industry_classification']
for column in categorical_columns:
    data_emp[column] = data_emp[column].astype(str).str.lower()

# Convert 'REF_DATE' to datetime format
data_emp['REF_DATE'] = pd.to_datetime(data_emp['REF_DATE'], format='%Y')

# Ensure all categorical variables are type 'category'
for column in categorical_columns:
    data_emp[column] = data_emp[column].astype('category')

# Ensure 'VALUE' is numeric
data_emp['VALUE'] = pd.to_numeric(data_emp['VALUE'], errors='coerce')

# Create a separate DataFrame for rows with missing data for later assessment
data_missing = data_emp[data_emp.isnull().any(axis=1)]

# Remove rows with any missing/NA/NAN/NULL data from the working dataset
data_emp = data_emp.dropna()


# Visualize the outliers using a box plot before filtering
before_outliers = plt.figure(figsize=(12, 6))
sns.boxplot(data=data_emp, y='VALUE', palette='viridis')
plt.title('Box Plot of VALUE to Visualize Data With Outliers')
plt.show()

# Identifying and Correcting Anomalies/Outliers
# For continuous data, using the IQR method to detect and remove outliers
Q1 = data_emp['VALUE'].quantile(0.25)
Q3 = data_emp['VALUE'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out the outliers
data_emp_clean = data_emp[(data_emp['VALUE'] >= lower_bound) & (data_emp['VALUE'] <= upper_bound)]

# Visualize the outliers using a box plot before filtering
after_outliers = plt.figure(figsize=(12, 6))
sns.boxplot(data=data_emp_clean, y='VALUE', palette='viridis')
plt.title('Box Plot of VALUE to Visualize Data Without Outliers')
plt.show()

# Print the shape of the dataset after removing outliers
print('Data Shape Before Removing Outliers:', data_emp.shape)
print('Data Shape After Removing Outliers:', data_emp_clean.shape)

# Calculate descriptive statistics for employment ('VALUE') by 'NAICS' and 'REF_DATE'
emp_by_naics_ref_date = data_emp_clean.groupby(['NAICS', 'REF_DATE'], observed=True)['VALUE'].describe()
print('Descriptive Statistics for Employment by NAICS and REF_DATE:')
print(emp_by_naics_ref_date.head())

# Calculate descriptive statistics for employment ('VALUE') by 'NAICS'  and 'Age group'
wages_by_naics_age_group = data_emp_clean.groupby(['NAICS', 'Age group'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Employment by NAICS and Age Group:')
print(wages_by_naics_age_group.head())

# Calculate descriptive statistics for employment ('VALUE') by 'NAICS' and 'Sex'
wages_by_naics_sex = data_emp_clean.groupby(['NAICS', 'Sex'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Employment by NAICS and Sex:')
print(wages_by_naics_sex.head())

# Calculate descriptive statistics for employment ('VALUE') by 'NAICS' and 'GEO'
wages_by_naics_geo = data_emp_clean.groupby(['NAICS', 'GEO'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Employment by NAICS and Geographic Location:')
print(wages_by_naics_geo.head())

# Calculate descriptive statistics for employment ('VALUE') by 'REF_DATE' and 'Age group' for January of each year
data_emp_clean['REF_DATE'] = pd.to_datetime(data_emp_clean['REF_DATE'])
data_january = data_emp_clean[data_emp_clean['REF_DATE'].dt.month == 1]

# Descriptive statistics for wages ('VALUE') by 'Age group' for January of each year
wages_by_age_group_january = data_january.groupby(['REF_DATE', 'Age group'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Wages by Age Group for January of Each Year:')
print(wages_by_age_group_january.head())

# Descriptive statistics for wages ('VALUE') by 'Sex' for January of each year
wages_by_sex_january = data_january.groupby(['REF_DATE', 'Sex'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Wages by Sex for January of Each Year:')
print(wages_by_sex_january.head())

# Descriptive statistics for wages ('VALUE') by 'GEO' for January of each year
wages_by_geo_january = data_january.groupby(['REF_DATE', 'GEO'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Wages by Geographic Location for January of Each Year:')
print(wages_by_geo_january.head())

# Descriptive statistics for wages ('VALUE') by 'industry_classification' for January of each year
wages_by_industry_classification_january = data_january.groupby(['REF_DATE', 'industry_classification'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Wages by Industry Classification for January of Each Year:')
print(wages_by_industry_classification_january.head())

# Descriptive statistics for wages ('VALUE') by 'NAICS' for January of each year
wages_by_naics_january = data_january.groupby(['REF_DATE', 'NAICS'], observed=True)['VALUE'].describe()
print('\nDescriptive Statistics for Wages by NAICS for January of Each Year:')
print(wages_by_naics_january.head())
