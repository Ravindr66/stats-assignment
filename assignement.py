# =============================================================
# Statistics and Trends Assignment
# Dataset: Google Play Store Apps
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# =============================================================
# STEP 1: LOAD AND CLEAN THE DATA
# =============================================================

# Load the dataset
df = pd.read_csv('googleplaystore.csv')

# Show basic info
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Clean the data
# Remove duplicate rows
df = df.drop_duplicates()

# Clean 'Rating' column - keep only valid ratings between 0 and 5
df = df[pd.to_numeric(df['Rating'], errors='coerce').between(0, 5)]
df['Rating'] = df['Rating'].astype(float)

# Clean 'Reviews' column - remove non-numeric values
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df = df.dropna(subset=['Reviews'])
df['Reviews'] = df['Reviews'].astype(int)

# Clean 'Installs' column - remove + and , symbols
df['Installs'] = df['Installs'].str.replace('+', '', regex=False)
df['Installs'] = df['Installs'].str.replace(',', '', regex=False)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df = df.dropna(subset=['Installs'])

print("\nCleaned dataset shape:", df.shape)

# =============================================================
# STEP 2: FOUR STATISTICAL MOMENTS
# =============================================================

print("\n========== STATISTICAL MOMENTS FOR RATING ==========")

rating = df['Rating'].dropna()

mean_val     = rating.mean()
var_val      = rating.var()
skew_val     = skew(rating)
kurt_val     = kurtosis(rating)

print(f"1st Moment - Mean:     {mean_val:.4f}")
print(f"2nd Moment - Variance: {var_val:.4f}")
print(f"3rd Moment - Skewness: {skew_val:.4f}")
print(f"4th Moment - Kurtosis: {kurt_val:.4f}")

print("\nInterpretation:")
print(f"  Mean rating is {mean_val:.2f} out of 5, indicating generally positive reviews.")
print(f"  Variance of {var_val:.2f} shows ratings are fairly consistent.")
print(f"  Skewness of {skew_val:.2f} means the distribution is slightly left-skewed (most apps rated high).")
print(f"  Kurtosis of {kurt_val:.2f} indicates the distribution has heavy tails compared to normal.")

# =============================================================
# STEP 3: RELATIONAL PLOT - Scatter plot (Reviews vs Rating)
# =============================================================

plt.figure(figsize=(10, 6))
plt.scatter(df['Reviews'], df['Rating'], alpha=0.3, color='steelblue', edgecolors='none')
plt.xscale('log')  # log scale because reviews vary hugely
plt.title('Relationship Between Reviews and Rating', fontsize=14)
plt.xlabel('Number of Reviews (log scale)', fontsize=12)
plt.ylabel('App Rating', fontsize=12)
plt.tight_layout()
plt.savefig('relational_plot.png')
plt.show()
print("Relational plot saved as relational_plot.png")

# =============================================================
# STEP 4: CATEGORICAL PLOT - Bar chart (Apps per Category)
# =============================================================

plt.figure(figsize=(14, 7))
category_counts = df['Category'].value_counts().head(10)  # top 10 categories
category_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Top 10 App Categories on Google Play Store', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Number of Apps', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('categorical_plot.png')
plt.show()
print("Categorical plot saved as categorical_plot.png")

# =============================================================
# STEP 5: STATISTICAL PLOT - Correlation Heatmap
# =============================================================

plt.figure(figsize=(8, 6))
numeric_cols = df[['Rating', 'Reviews', 'Installs']].corr()
sns.heatmap(numeric_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap: Rating, Reviews and Installs', fontsize=14)
plt.tight_layout()
plt.savefig('statistical_plot.png')
plt.show()
print("Statistical plot saved as statistical_plot.png")

print("\n✅ All done! Check your stats_assignment folder for the 3 plot images.")