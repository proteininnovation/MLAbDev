

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind, spearmanr

# User-configurable parameters
csv_file = "/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/transform_onehot_sample_size_performance_psr_filter.csv"
#csv_file = "/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/transform_onehot_ipi_elisa_ngs_sample_size_performance_psr_filter2.csv"  # Your 12k dataset (update as needed)
smoothing_window = 5  # Adjust for rolling average (e.g., 3=more detail, 10=smoother trends)

# Bucket configuration for <12k dataset (customize thresholds here)
low_threshold = 1000
mid_threshold = 5000

low_threshold = 5000
mid_threshold = 175000

# Read the CSV data from the file
df = pd.read_csv(csv_file)

# Display basic statistics
print("Descriptive Statistics:")
print(df.describe())

# Compute Pearson correlations with sample_size
pearson_correlations = df.corr(method='pearson')['sample_size'].drop('sample_size')
print("\nPearson Correlations with Sample Size:")
print(pearson_correlations)

# Compute Spearman correlations (non-parametric)
spearman_corrs = {}
for col in ['auc', 'accuracy', 'f1_score', 'precision', 'recall']:
    corr, pval = spearmanr(df['sample_size'], df[col])
    spearman_corrs[col] = corr
print("\nSpearman Correlations with Sample Size:")
print(pd.Series(spearman_corrs))

# Bucketed averages for AUC
buckets = {
    f'Low (<={low_threshold})': df[df['sample_size'] <= low_threshold]['auc'],
    f'Mid ({low_threshold}-{mid_threshold})': 
        df[(df['sample_size'] > low_threshold) & (df['sample_size'] <= mid_threshold)]['auc'],
    f'High (>{mid_threshold})': df[df['sample_size'] > mid_threshold]['auc']
}
bucket_means = {k: v.mean() for k, v in buckets.items()}
bucket_stds = {k: v.std() for k, v in buckets.items()}
print("\nBucketed AUC Averages:")
for k, mean in bucket_means.items():
    print(f"{k}: {mean:.4f}")

# Updated T-tests (dynamic between consecutive buckets)
bucket_list = list(buckets.keys())
print("\nT-Tests between consecutive buckets:")
for i in range(len(bucket_list) - 1):
    name1 = bucket_list[i]
    name2 = bucket_list[i + 1]
    t_stat, p_val = ttest_ind(buckets[name1], buckets[name2], equal_var=False)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{name1} vs {name2}: t={t_stat:.2f}, p={p_val:.4f} {sig}")

# Calculate consecutive differences for AUC
df['auc_diff'] = df['auc'].diff()
print("\nConsecutive AUC Differences:")
print(df[['sample_size', 'auc', 'auc_diff']])

# Linear and Polynomial Regression for each metric vs. sample_size
regression_results = []
poly_regression_results = []
for metric in ['auc', 'accuracy', 'f1_score', 'precision', 'recall']:
    # Linear Regression
    X_lin = sm.add_constant(df['sample_size'])
    y = df[metric]
    model_lin = sm.OLS(y, X_lin).fit()
    slope_lin = model_lin.params[1]
    intercept_lin = model_lin.params[0]
    p_value_lin = model_lin.pvalues[1]
    r_squared_lin = model_lin.rsquared
    regression_results.append({
        'metric': metric,
        'type': 'linear',
        'slope': slope_lin,
        'intercept': intercept_lin,
        'p_value': p_value_lin,
        'r_squared': r_squared_lin
    })
    print(f"\nLinear Regression for {metric} vs. Sample Size:")
    print(model_lin.summary())

    # Polynomial Regression (add quadratic term for non-linearity)
    df['sample_size_sq'] = df['sample_size'] ** 2  # Add polynomial term
    X_poly = sm.add_constant(df[['sample_size', 'sample_size_sq']])
    model_poly = sm.OLS(y, X_poly).fit()
    slope_poly = model_poly.params[1]  # Slope for linear term
    quad_coeff = model_poly.params[2]  # Coefficient for quadratic term (if negative, indicates tapering)
    p_value_poly_lin = model_poly.pvalues[1]
    p_value_poly_quad = model_poly.pvalues[2]
    r_squared_poly = model_poly.rsquared
    poly_regression_results.append({
        'metric': metric,
        'type': 'polynomial',
        'intercept': model_poly.params[0],
        'slope_linear': slope_poly,
        'coeff_quadratic': quad_coeff,
        'p_value_linear': p_value_poly_lin,
        'p_value_quadratic': p_value_poly_quad,
        'r_squared': r_squared_poly
    })
    print(f"\nPolynomial Regression for {metric} vs. Sample Size (with Quadratic Term):")
    print(model_poly.summary())

# Save regression results to CSVs
reg_df = pd.DataFrame(regression_results)
reg_df.to_csv(f"{csv_file}.linear_regression_results.csv", index=False)
poly_reg_df = pd.DataFrame(poly_regression_results)
poly_reg_df.to_csv(f"{csv_file}.polynomial_regression_results.csv"  , index=False)
print("\nRegression results saved to 'linear_regression_results.csv' and 'polynomial_regression_results.csv'")

# Visualization: Line plot with rolling averages and regression trendlines
plt.figure(figsize=(6, 4))
metrics = ['auc', 'accuracy', 'f1_score', 'precision', 'recall']
markers = ['o', 's', '^', 'd', 'v']
for i, metric in enumerate(metrics):
    # Rolling average for smoothing
    df[f'{metric}_rolling'] = df[metric].rolling(window=smoothing_window, min_periods=1).mean()
    sns.lineplot(data=df, x='sample_size', y=f'{metric}_rolling', marker=markers[i], label=metric.capitalize(), alpha=0.7)
    
    # Linear regression trendline (dashed gray)
    filtered_reg = reg_df[(reg_df['metric'] == metric) & (reg_df['type'] == 'linear')]
    if not filtered_reg.empty:
        reg = filtered_reg.iloc[0]
        if reg['p_value'] < 0.05:
            trend_lin = reg['intercept'] + reg['slope'] * df['sample_size']
            plt.plot(df['sample_size'], trend_lin, linestyle='--', color='gray', alpha=0.5, label=f'{metric} Linear Trend (p<0.05)' if i == 0 else None)

    # Polynomial trendline (dotted black, if quadratic significant)
    filtered_poly = poly_reg_df[poly_reg_df['metric'] == metric]
    if not filtered_poly.empty:
        poly_reg = filtered_poly.iloc[0]
        if poly_reg['p_value_quadratic'] < 0.05:
            trend_poly = poly_reg['intercept'] + \
                         poly_reg['slope_linear'] * df['sample_size'] + \
                         poly_reg['coeff_quadratic'] * df['sample_size_sq']
            plt.plot(df['sample_size'], trend_poly, linestyle=':', color='black', alpha=0.5, label=f'{metric} Poly Trend (p<0.05)' if i == 0 else None)
    else:
        print(f"Warning: No polynomial regression data for metric '{metric}' - skipping trendline.")    

# Add bucket means with error bars
bucket_centers = [low_threshold / 2, (low_threshold + mid_threshold) / 2, mid_threshold + 1000]  # Adjusted centers for <12k
plt.errorbar(bucket_centers, list(bucket_means.values()), yerr=list(bucket_stds.values()), fmt='x', color='black', label='Bucket Means ± Std', capsize=5)

plt.title('Model Performance Metrics vs. Trainning Set Size (Smoothed with Regression Trends)', fontsize=10)
plt.xlabel('Training Set Size')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=8, title_fontsize=10)

plt.savefig(f"{csv_file}.performance_vs_sample_size_enhanced.png")  # Save to file
plt.show()

# Heatmap of correlations (Pearson)
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Heatmap of Metrics')
plt.tight_layout()
plt.savefig(f"{csv_file}.correlation_heatmap.png")  # Save to file
plt.show()

# Scaling Efficiency Check (for memorization insight) - Updated to include polynomial info
print("\nScaling Efficiency (Generalization Check):")
for i, res in enumerate(regression_results):
    if i < len(poly_regression_results):  # Check bounds to avoid IndexError
        poly_res = poly_regression_results[i]
        status = "Positive scaling" if (res['slope'] > 0 and res['p_value'] < 0.05) else "No/Weak scaling"
        quad_note = f" (Quadratic tapering: coeff={poly_res['coeff_quadratic']:.2e}, p={poly_res['p_value_quadratic']:.4f})" if poly_res['p_value_quadratic'] < 0.05 else ""
        risk = " - Low memorization risk." if "Positive" in status else " - Potential plateau or memorization."
        print(f"{res['metric']}: {status} (slope={res['slope']:.6e}, p={res['p_value']:.4f}){quad_note}{risk}")
    else:
        print(f"Warning: No polynomial data for index {i} - skipping for {res['metric']}")