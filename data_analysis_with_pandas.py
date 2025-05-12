
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].apply(lambda x: iris.target_names[x])
except Exception as e:
    print("Error loading dataset:", e)

# Display first 5 rows
print(df.head())

# Check data types and missing values
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Basic statistics
print("\nDescriptive Statistics:\n", df.describe())

# Group by species and calculate mean of each numerical column
grouped_means = df.groupby("species").mean()
print("\nMean values per species:\n", grouped_means)

# Observations
for species in grouped_means.index:
    print(f"\nAverage petal length of {species}: {grouped_means.loc[species, 'petal length (cm)']:.2f}")

# 1. Line Chart - Simulated trend using cumulative sum
df['index'] = df.index
plt.figure(figsize=(10,6))
for species in df['species'].unique():
    temp = df[df['species'] == species]
    plt.plot(temp['index'], temp['sepal length (cm)'].cumsum(), label=species)
plt.title('Cumulative Sepal Length over Index')
plt.xlabel('Index')
plt.ylabel('Cumulative Sepal Length')
plt.legend()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(8,6))
sns.barplot(x=grouped_means.index, y=grouped_means['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(8,6))
plt.hist(df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
