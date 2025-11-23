"""
Quick Data Exploration Script
Visualize and understand your Fe-Ni alloy data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("  Fe-Ni Alloy Data Explorer")
print("=" * 70)

# Load data
df = pd.read_csv('fe_ni_alloys.csv')

print(f"\n✓ Loaded {len(df)} materials")
print("\n" + "=" * 70)
print("Dataset Overview")
print("=" * 70)
print(df.info())

print("\n" + "=" * 70)
print("First 5 Materials")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("Statistical Summary")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("Composition Analysis")
print("=" * 70)
print(f"Fe fraction range: {df['fe_fraction'].min():.3f} - {df['fe_fraction'].max():.3f}")
print(f"Ni fraction range: {df['ni_fraction'].min():.3f} - {df['ni_fraction'].max():.3f}")

print("\n" + "=" * 70)
print("Stability Analysis")
print("=" * 70)
stable_count = df['is_stable'].sum()
near_stable = (df['energy_above_hull'] < 0.1).sum()
print(f"Stable materials: {stable_count} ({stable_count/len(df)*100:.1f}%)")
print(f"Near-stable (E_hull < 0.1 eV): {near_stable} ({near_stable/len(df)*100:.1f}%)")
print(f"Average E_hull: {df['energy_above_hull'].mean():.4f} eV/atom")

print("\n" + "=" * 70)
print("Creating Visualizations...")
print("=" * 70)

# Create comprehensive visualization
fig = plt.figure(figsize=(15, 10))

# 1. Formation energy vs composition
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(df['ni_fraction'], df['formation_energy'],
                      c=df['energy_above_hull'], s=100, cmap='RdYlGn_r', alpha=0.7)
ax1.set_xlabel('Ni Fraction')
ax1.set_ylabel('Formation Energy (eV/atom)')
ax1.set_title('Formation Energy vs Ni Content')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='E above hull (eV)')

# 2. Density vs composition
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(df['ni_fraction'], df['density'], s=100, alpha=0.7, color='blue')
ax2.set_xlabel('Ni Fraction')
ax2.set_ylabel('Density (g/cm³)')
ax2.set_title('Density vs Ni Content')
ax2.grid(True, alpha=0.3)

# 3. Energy above hull distribution
ax3 = plt.subplot(2, 3, 3)
ax3.hist(df['energy_above_hull'], bins=15, edgecolor='black', alpha=0.7, color='purple')
ax3.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Synthesizable limit')
ax3.set_xlabel('Energy Above Hull (eV/atom)')
ax3.set_ylabel('Count')
ax3.set_title('Stability Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Formation energy distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(df['formation_energy'], bins=15, edgecolor='black', alpha=0.7, color='green')
ax4.set_xlabel('Formation Energy (eV/atom)')
ax4.set_ylabel('Count')
ax4.set_title('Formation Energy Distribution')
ax4.grid(True, alpha=0.3)

# 5. Density vs formation energy
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(df['formation_energy'], df['density'], s=100, alpha=0.7, color='orange')
ax5.set_xlabel('Formation Energy (eV/atom)')
ax5.set_ylabel('Density (g/cm³)')
ax5.set_title('Density vs Formation Energy')
ax5.grid(True, alpha=0.3)

# 6. Crystal system distribution
ax6 = plt.subplot(2, 3, 6)
if 'crystal_system' in df.columns:
    crystal_counts = df['crystal_system'].value_counts()
    ax6.bar(range(len(crystal_counts)), crystal_counts.values, alpha=0.7, color='teal')
    ax6.set_xticks(range(len(crystal_counts)))
    ax6.set_xticklabels(crystal_counts.index, rotation=45, ha='right')
    ax6.set_ylabel('Count')
    ax6.set_title('Crystal System Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
else:
    ax6.text(0.5, 0.5, 'Crystal system\ndata not available',
             ha='center', va='center', fontsize=12)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('fe_ni_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved comprehensive analysis to: fe_ni_comprehensive_analysis.png")

# Correlation analysis
print("\n" + "=" * 70)
print("Correlation Analysis")
print("=" * 70)

numeric_cols = ['fe_fraction', 'ni_fraction', 'formation_energy',
                'energy_above_hull', 'density', 'volume_per_atom']
available_cols = [col for col in numeric_cols if col in df.columns]

if len(available_cols) > 2:
    correlation = df[available_cols].corr()
    print(correlation)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Property Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fe_ni_correlations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved correlation heatmap to: fe_ni_correlations.png")

print("\n" + "=" * 70)
print("Key Insights")
print("=" * 70)

# Find most stable material
most_stable = df.loc[df['energy_above_hull'].idxmin()]
print(f"\nMost Stable Material:")
print(f"  Formula: {most_stable['formula']}")
print(f"  Energy above hull: {most_stable['energy_above_hull']:.4f} eV/atom")
print(f"  Formation energy: {most_stable['formation_energy']:.4f} eV/atom")
print(f"  Density: {most_stable['density']:.2f} g/cm³")

# Find highest density
highest_density = df.loc[df['density'].idxmax()]
print(f"\nHighest Density Material:")
print(f"  Formula: {highest_density['formula']}")
print(f"  Density: {highest_density['density']:.2f} g/cm³")
print(f"  Ni fraction: {highest_density['ni_fraction']:.3f}")

print("\n" + "=" * 70)
print("Exploration Complete!")
print("=" * 70)
print("\nGenerated files:")
print("  1. fe_ni_comprehensive_analysis.png")
print("  2. fe_ni_correlations.png")
print("\nNext steps:")
print("  - Train ML models: python train_baseline_model.py")
print("  - Download more data: python mp_data_download.py")
print("  - Review project plan for Phase 1 tasks")
print("=" * 70)
