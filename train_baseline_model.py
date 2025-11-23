"""
Baseline ML Model - Predict Formation Energy from Composition
Simple example using Fe-Ni data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

print("=" * 70)
print("  Baseline ML Model: Predicting Formation Energy")
print("=" * 70)

# Load data
print("\nLoading Fe-Ni alloy data...")
df = pd.read_csv('fe_ni_alloys.csv')
print(f"Loaded {len(df)} materials")

# Prepare features and target
print("\nPreparing features...")
# Features: composition fractions
X = df[['fe_fraction', 'ni_fraction']].values
# Target: formation energy
y = df['formation_energy'].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train models
print("\n" + "=" * 70)
print("Training Models...")
print("=" * 70)

# Model 1: Linear Regression
print("\n1. Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)

print(f"   Train R²: {lr_train_r2:.4f}")
print(f"   Test R²:  {lr_test_r2:.4f}")
print(f"   Test MAE: {lr_test_mae:.4f} eV/atom")

# Model 2: Random Forest
print("\n2. Random Forest")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)

print(f"   Train R²: {rf_train_r2:.4f}")
print(f"   Test R²:  {rf_test_r2:.4f}")
print(f"   Test MAE: {rf_test_mae:.4f} eV/atom")

# Visualize results
print("\n" + "=" * 70)
print("Creating Visualizations...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression plot
axes[0].scatter(y_test, lr_test_pred, alpha=0.6, s=100)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Formation Energy (eV/atom)')
axes[0].set_ylabel('Predicted Formation Energy (eV/atom)')
axes[0].set_title(f'Linear Regression (R² = {lr_test_r2:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest plot
axes[1].scatter(y_test, rf_test_pred, alpha=0.6, s=100, color='green')
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Formation Energy (eV/atom)')
axes[1].set_ylabel('Predicted Formation Energy (eV/atom)')
axes[1].set_title(f'Random Forest (R² = {rf_test_r2:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_model_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to: baseline_model_results.png")

# Feature importance (Random Forest)
print("\n" + "=" * 70)
print("Feature Importance (Random Forest):")
print("=" * 70)
feature_names = ['Fe Fraction', 'Ni Fraction']
importances = rf_model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\nDataset: {len(df)} Fe-Ni alloys")
print(f"Features: Composition (Fe, Ni fractions)")
print(f"Target: Formation Energy")
print(f"\nBest Model: Random Forest")
print(f"  Test R²: {rf_test_r2:.4f}")
print(f"  Test MAE: {rf_test_mae:.4f} eV/atom")

print("\n" + "=" * 70)
print("Next Steps:")
print("=" * 70)
print("1. Download more alloy systems for bigger dataset")
print("2. Add more features (density, volume, etc.)")
print("3. Try advanced models (XGBoost, Neural Networks)")
print("4. Implement cross-validation")
print("5. Add uncertainty quantification")
print("\n" + "=" * 70)
