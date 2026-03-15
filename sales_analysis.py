# ============================================================
#  Sales Data Analysis & Visualization Dashboard
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ============================================================
# 1. GENERATE / LOAD DATA
#    Replace the block below with:  df = pd.read_csv("your_file.csv")
# ============================================================
np.random.seed(42)
n = 50_000

categories   = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books"]
regions      = ["North", "South", "East", "West"]
months       = pd.date_range("2021-01-01", periods=36, freq="MS")

df = pd.DataFrame({
    "date"      : np.random.choice(months, n),
    "category"  : np.random.choice(categories, n),
    "region"    : np.random.choice(regions, n),
    "units_sold": np.random.randint(1, 50, n),
    "unit_price": np.round(np.random.uniform(5, 500, n), 2),
})
df["revenue"] = df["units_sold"] * df["unit_price"]
print(f"Dataset loaded: {len(df):,} rows\n")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("── Basic Info ──────────────────────────────────────")
print(df.describe())

# Monthly revenue trend
monthly = (
    df.groupby("date")["revenue"]
    .sum()
    .reset_index()
    .rename(columns={"revenue": "monthly_revenue"})
)

# Revenue by category
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)

# Revenue by region
reg_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

# ============================================================
# 3. VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Sales Data Analysis Dashboard", fontsize=16, fontweight="bold", y=1.01)

# -- 3a. Monthly Revenue Trend --
axes[0, 0].plot(monthly["date"], monthly["monthly_revenue"], marker="o", linewidth=2, color="#2196F3")
axes[0, 0].fill_between(monthly["date"], monthly["monthly_revenue"], alpha=0.15, color="#2196F3")
axes[0, 0].set_title("Monthly Revenue Trend")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Revenue ($)")
axes[0, 0].tick_params(axis="x", rotation=45)

# -- 3b. Revenue by Category --
sns.barplot(x=cat_rev.values, y=cat_rev.index, ax=axes[0, 1], palette="Blues_d")
axes[0, 1].set_title("Revenue by Category")
axes[0, 1].set_xlabel("Total Revenue ($)")
axes[0, 1].set_ylabel("")

# -- 3c. Revenue by Region --
axes[1, 0].pie(
    reg_rev.values,
    labels=reg_rev.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("pastel"),
)
axes[1, 0].set_title("Revenue Share by Region")

# -- 3d. Revenue Distribution --
sns.histplot(df["revenue"], bins=60, kde=True, ax=axes[1, 1], color="#4CAF50")
axes[1, 1].set_title("Revenue Distribution per Transaction")
axes[1, 1].set_xlabel("Revenue ($)")

plt.tight_layout()
plt.savefig("dashboard.png", bbox_inches="tight")
plt.show()
print("\n✅ Dashboard saved as dashboard.png")

# ============================================================
# 4. REGRESSION — Monthly Sales Forecasting
# ============================================================
print("\n── Sales Forecasting (Linear Regression) ──────────")

monthly["month_index"] = np.arange(len(monthly))

X = monthly[["month_index"]]
y = monthly["monthly_revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MAE : ${mae:,.2f}")
print(f"R²  : {r2:.4f}  ({r2*100:.1f}% accuracy)")

# -- Forecast next 6 months --
future_idx = np.arange(len(monthly), len(monthly) + 6).reshape(-1, 1)
future_rev = model.predict(future_idx)

print("\n── 6-Month Revenue Forecast ────────────────────────")
for i, rev in enumerate(future_rev, 1):
    print(f"  Month +{i}: ${rev:,.2f}")

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(monthly["month_index"], y, label="Actual", marker="o", color="#2196F3")
plt.plot(X_test["month_index"], y_pred, label="Test Prediction", linestyle="--", color="#FF9800")
plt.plot(future_idx, future_rev, label="Forecast", linestyle="--", marker="s", color="#F44336")
plt.title("Monthly Revenue — Actual vs Forecast")
plt.xlabel("Month Index")
plt.ylabel("Revenue ($)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png", bbox_inches="tight")
plt.show()
print("✅ Forecast plot saved as forecast.png")

# ============================================================
# 5. KEY INSIGHTS
# ============================================================
print("\n── Key Insights ────────────────────────────────────")
print(f"  Top category  : {cat_rev.idxmax()} (${cat_rev.max():,.2f})")
print(f"  Top region    : {reg_rev.idxmax()} (${reg_rev.max():,.2f})")
print(f"  Avg monthly revenue : ${monthly['monthly_revenue'].mean():,.2f}")
print(f"  Peak month    : {monthly.loc[monthly['monthly_revenue'].idxmax(), 'date'].strftime('%B %Y')}")
print("\nAnalysis complete ✅")
