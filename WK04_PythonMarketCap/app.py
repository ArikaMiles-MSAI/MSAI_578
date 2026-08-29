"""
Python script to analyze Market Capitalization vs. Profit by Sector
from Fortune500Sector.xlsx sample of Fortune 500 firms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
mc_df = pd.read_excel('Fortune500Sector.xlsx')

# Standardize column names for internal use
mc_df = mc_df.rename(columns={
    'Profits ($ millions)': 'Profit',
    'Market Capitalization ($ millions)': 'MarketCap'
})

# -----------------------------------------------------------------------------
# Objective 1: Scatter chart colored by industry sector
# Market Capitalization (vertical/y) vs Profit (horizontal/x)
# -----------------------------------------------------------------------------
mc_sectors = mc_df['Sector'].unique()
mc_colors = plt.cm.tab10(np.linspace(0, 1, len(mc_sectors)))
mc_color_map = dict(zip(mc_sectors, mc_colors))

mc_fig1, mc_ax1 = plt.subplots(figsize=(10, 7))
for mc_sector in mc_sectors:
    mc_subset = mc_df[mc_df['Sector'] == mc_sector]
    mc_ax1.scatter(
        mc_subset['Profit'],
        mc_subset['MarketCap'],
        c=[mc_color_map[mc_sector]],
        label=mc_sector,
        s=80,
        edgecolors='k',
        alpha=0.8
    )

mc_ax1.set_xlabel('Profit ($ millions)', fontsize=12)
mc_ax1.set_ylabel('Market Capitalization ($ millions)', fontsize=12)
mc_ax1.set_title('Market Capitalization vs. Profit by Sector', fontsize=14)
mc_ax1.legend(title='Sector', loc='upper left')
mc_ax1.grid(True, alpha=0.3)
plt.tight_layout()
mc_fig1.savefig('scatter_by_sector.png', dpi=150, bbox_inches='tight')
plt.close(mc_fig1)

# -----------------------------------------------------------------------------
# Objective 2: Emphasize Healthcare sector
# Other sectors: gray, no fill (facecolors='none')
# Trendline based only on Healthcare observations
# -----------------------------------------------------------------------------
mc_fig2, mc_ax2 = plt.subplots(figsize=(10, 7))

# Non-healthcare observations: gray edges, no fill
mc_non_hc = mc_df[mc_df['Sector'] != 'Healthcare']
mc_ax2.scatter(
    mc_non_hc['Profit'],
    mc_non_hc['MarketCap'],
    facecolors='none',
    edgecolors='gray',
    s=80,
    alpha=0.7,
    label='Other Sectors'
)

# Healthcare observations: solid red fill
mc_hc = mc_df[mc_df['Sector'] == 'Healthcare']
mc_ax2.scatter(
    mc_hc['Profit'],
    mc_hc['MarketCap'],
    c='tab:red',
    s=100,
    edgecolors='k',
    alpha=0.9,
    label='Healthcare'
)

# Ordinary least squares (OLS) linear trendline via numpy.polyfit (degree 1)
mc_X = mc_hc['Profit'].values
mc_y = mc_hc['MarketCap'].values
mc_coeffs = np.polyfit(mc_X, mc_y, 1)
mc_slope = mc_coeffs[0]
mc_intercept = mc_coeffs[1]

# Compute R-squared for the Healthcare fit
mc_y_pred = mc_slope * mc_X + mc_intercept
mc_ss_res = np.sum((mc_y - mc_y_pred) ** 2)
mc_ss_tot = np.sum((mc_y - np.mean(mc_y)) ** 2)
mc_r2 = 1 - (mc_ss_res / mc_ss_tot)

# Plot the trendline across the observed Profit range of Healthcare firms
mc_x_line = np.linspace(mc_hc['Profit'].min(), mc_hc['Profit'].max(), 100)
mc_y_line = mc_slope * mc_x_line + mc_intercept
mc_ax2.plot(
    mc_x_line,
    mc_y_line,
    color='darkred',
    linewidth=2,
    label=f'Healthcare Trendline (R²={mc_r2:.3f})'
)

mc_ax2.set_xlabel('Profit ($ millions)', fontsize=12)
mc_ax2.set_ylabel('Market Capitalization ($ millions)', fontsize=12)
mc_ax2.set_title(
    'Market Capitalization vs. Profit: Healthcare Emphasis with Trendline',
    fontsize=14
)
mc_ax2.legend(loc='upper left')
mc_ax2.grid(True, alpha=0.3)
plt.tight_layout()
mc_fig2.savefig('scatter_healthcare_trend.png', dpi=150, bbox_inches='tight')
plt.close(mc_fig2)

# Console summary of Healthcare fit for verification
print(f'Healthcare OLS slope (MarketCap per unit Profit): {mc_slope:.4f}')
print(f'Healthcare OLS intercept: {mc_intercept:.2f}')
print(f'Healthcare R-squared: {mc_r2:.4f}')
print(f'Healthcare Pearson correlation: {np.corrcoef(mc_X, mc_y)[0, 1]:.4f}')
print('Healthcare firms:')
print(mc_hc[['Company', 'Profit', 'MarketCap']].to_string(index=False))
