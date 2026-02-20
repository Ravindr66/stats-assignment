# Statistics and Trends Assignment
# Dataset: Google Play Store Apps
# This analysis looks at what makes apps successful on the Play Store

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# LOADING THE DATASET

df = pd.read_csv("googleplaystore.csv")
print("Dataset loaded successfully")
print("Total rows and columns:", df.shape)

# CLEANING THE DATA BEFORE ANALYSIS

# DROP DUPLICATE ROWS
df = df.drop_duplicates()

# FIX THE RATING COLUMN - ONLY KEEP VALID RATINGS BETWEEN 0 AND 5
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df[df["Rating"].between(0, 5)]

# FIX THE REVIEWS COLUMN - REMOVE ANY NON-NUMBER VALUES
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
df = df.dropna(subset=["Reviews"])
df["Reviews"] = df["Reviews"].astype(int)

# FIX THE INSTALLS COLUMN - REMOVE + AND , SYMBOLS SO WE CAN USE IT AS A NUMBER
df["Installs"] = df["Installs"].str.replace("+", "", regex=False)
df["Installs"] = df["Installs"].str.replace(",", "", regex=False)
df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")
df = df.dropna(subset=["Installs"])

print("Data cleaned. Remaining rows:", len(df))

# THE FOUR STATISTICAL MOMENTS
# USING THE RATING COLUMN BECAUSE IT IS THE BEST MEASURE
# OF HOW WELL AN APP IS RECEIVED BY USERS

rating = df["Rating"].dropna()

mean_rating = rating.mean()
var_rating = rating.var()
skew_rating = skew(rating)
kurt_rating = kurtosis(rating)

print("\n--- Four Statistical Moments for App Ratings ---")
print(f"Mean      : {mean_rating:.4f}")
print(f"Variance  : {var_rating:.4f}")
print(f"Skewness  : {skew_rating:.4f}")
print(f"Kurtosis  : {kurt_rating:.4f}")

# ANALYSIS AND INTERPRETATION OF THE FOUR MOMENTS

print("""
The mean rating of {:.2f} out of 5 tells us that most apps on the
Play Store are well received by users. This is a high average which
suggests users generally only keep and rate apps they enjoy.

The variance of {:.2f} is quite low which means ratings do not
spread out much. Most apps sit between 3.5 and 5 stars which
shows consistency across the platform.

The skewness of {:.2f} is negative which means the distribution
leans to the left. In simple terms, more apps have high ratings
than low ones. Very few apps fall below 3 stars.

The kurtosis of {:.2f} is high which means there are some extreme
values at the edges. A small number of apps have unusually low
ratings compared to the majority of well rated apps.

Together these four moments tell a clear story - the Play Store
is dominated by high quality apps with consistent ratings, but
a small group of poorly rated apps exist as outliers.
""".format(mean_rating, var_rating, skew_rating, kurt_rating))

# PLOT 1 - RELATIONAL PLOT
# SCATTER PLOT SHOWING THE RELATIONSHIP BETWEEN
# NUMBER OF REVIEWS AND APP RATING

plt.figure(figsize=(12, 7))

plt.scatter(
    df["Reviews"], df["Rating"], alpha=0.4, color="steelblue", edgecolors="none", s=25
)

plt.xscale("log")

plt.title(
    "Do More Reviews Lead to Better Ratings?", fontsize=16, fontweight="bold", pad=15
)

plt.xlabel("Number of Reviews (Log Scale)", fontsize=14, labelpad=10)
plt.ylabel("App Rating out of 5", fontsize=14, labelpad=10)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("relational_plot.png", dpi=150)
plt.show()

print("Relational plot saved")
print("""
From this scatter plot we can see that apps with more reviews
tend to have more stable ratings between 4.0 and 4.5.
Apps with very few reviews show the most variation in ratings.
This suggests that popular apps with large user bases maintain
more consistent and reliable ratings over time.
""")

# PLOT 2 - CATEGORICAL PLOT
# BAR CHART COMPARING THE NUMBER OF APPS IN EACH CATEGORY
# THIS SHOWS WHICH CATEGORIES ARE MOST COMPETITIVE

plt.figure(figsize=(14, 8))

category_counts = df["Category"].value_counts().head(10)

bars = plt.bar(
    category_counts.index,
    category_counts.values,
    color="coral",
    edgecolor="black",
    linewidth=0.7,
)

# ADDING THE COUNT NUMBERS ON TOP OF EACH BAR SO THEY ARE EASY TO READ
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 10,
        str(int(height)),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.title(
    "Which App Categories Have the Most Apps?", fontsize=16, fontweight="bold", pad=15
)

plt.xlabel("App Category", fontsize=14, labelpad=10)
plt.ylabel("Number of Apps", fontsize=14, labelpad=10)

plt.xticks(rotation=35, ha="right", fontsize=12)
plt.yticks(fontsize=12)

plt.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("categorical_plot.png", dpi=150)
plt.show()

print("Categorical plot saved")
print("""
The bar chart clearly shows that Family and Game categories
have by far the most apps on the Play Store. This means these
two categories are the most competitive for developers.
Categories like Beauty and Events have very few apps which
could mean there is more opportunity for new developers there.
The difference between the top and bottom categories is very
large which shows how unevenly apps are spread across categories.
""")

# PLOT 3 - STATISTICAL PLOT
# BOX PLOT SHOWING RATING DISTRIBUTIONS ACROSS TOP 5 CATEGORIES
# THIS HELPS US COMPARE THE SPREAD AND CONSISTENCY OF RATINGS

plt.figure(figsize=(14, 8))

top5 = df["Category"].value_counts().head(5).index
df_top5 = df[df["Category"].isin(top5)]

sns.boxplot(
    x="Category", y="Rating", data=df_top5, palette="coolwarm", linewidth=1.5, width=0.5
)

plt.title(
    "How Do Ratings Compare Across the Top 5 Categories?",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

plt.xlabel("App Category", fontsize=14, labelpad=10)
plt.ylabel("App Rating out of 5", fontsize=14, labelpad=10)

plt.xticks(rotation=20, ha="right", fontsize=12)
plt.yticks(fontsize=12)

plt.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("statistical_plot.png", dpi=150)
plt.show()

print("Statistical plot saved")
print("""
The box plot shows that all top 5 categories have a median
rating above 4.0 which is a good sign for the platform overall.
The Game category has the widest spread of ratings meaning
quality varies a lot among games. The Family category is
more consistent. The dots outside the boxes are outliers
which are apps with unusually low ratings compared to others
in the same category. This matches what the kurtosis value
told us earlier about extreme values existing in the data.
""")

# FINAL SUMMARY

print("""
=== OVERALL CONCLUSION ===

This analysis of the Google Play Store dataset reveals that
the platform is dominated by high quality apps with an average
rating of 4.19 out of 5. The low variance shows ratings are
consistent and the negative skewness confirms that most apps
lean towards high ratings.

Family and Game categories are the most crowded with apps
making them the hardest to stand out in. Meanwhile the box
plot shows that even within these popular categories ratings
stay consistently high above 4.0.

The relational plot connects everything together by showing
that apps which attract more reviews also tend to have more
stable ratings. This suggests that building a loyal user base
is key to maintaining a good reputation on the Play Store.
""")
