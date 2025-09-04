# Titanic Data Cleaning & EDA
# Install required libraries if not already installed:
# pip install pandas matplotlib seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset directly from seaborn
titanic = sns.load_dataset("titanic")

# Show first 5 rows
print("First 5 rows of dataset:")
print(titanic.head())

# Basic info
print("\nDataset Info:")
print(titanic.info())

print("\nMissing values count:")
print(titanic.isnull().sum())

# Drop columns with too many missing values
titanic_clean = titanic.drop(columns=["deck", "embark_town"])

# Fill missing values
titanic_clean["age"].fillna(titanic_clean["age"].median(), inplace=True)
titanic_clean["embarked"].fillna(titanic_clean["embarked"].mode()[0], inplace=True)

# Summary statistics
print("\nSummary Statistics:")
print(titanic_clean.describe())

# --- Exploratory Data Analysis (EDA) ---

# Survival countplot
plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=titanic_clean, palette="viridis")
plt.title("Survival Count")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="sex", hue="survived", data=titanic_clean, palette="mako")
plt.title("Survival by Gender")
plt.show()

# Survival by Class
plt.figure(figsize=(6,4))
sns.countplot(x="class", hue="survived", data=titanic_clean, palette="Set2")
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(titanic_clean["age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution of Passengers")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
# Select only numeric columns for correlation calculation
numeric_titanic = titanic_clean.select_dtypes(include=['number'])
sns.heatmap(numeric_titanic.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
