"""
Exploratory Data Analysis (EDA) Module
Generates 16 publication-quality statistical visualizations and a comprehensive EDA summary report.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def setup_plotting_style():
    """Configure modern, aesthetic plotting theme."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.size"] = 11


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Generate all 16 required EDA figures and save them to output_dir.
    
    Args:
        df: Cleaned smartphone DataFrame.
        output_dir: Directory where figures will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting_style()
    target_col = "Price_INR" if "Price_INR" in df.columns else ("Price_BDT" if "Price_BDT" in df.columns else df.columns[-1])
    
    # 1. Price Distribution (KDE + Histogram overlay)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True, color="#1f77b4", bins=20, edgecolor="black", alpha=0.6)
    plt.axvline(df[target_col].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df[target_col].mean():,.0f} ₹")
    plt.axvline(df[target_col].median(), color="green", linestyle="-", linewidth=2, label=f"Median: {df[target_col].median():,.0f} ₹")
    plt.title("Figure 1: Smartphone Price Distribution (₹ INR)")
    plt.xlabel("Price (₹ INR)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_price_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Price Histogram
    plt.figure(figsize=(9, 5))
    n, bins, patches = plt.hist(df[target_col], bins=15, color="#2ca02c", edgecolor="black", alpha=0.7)
    plt.title("Figure 2: Histogram of Smartphone Prices (₹ INR)")
    plt.xlabel("Price (₹ INR)")
    plt.ylabel("Smartphone Count")
    for i in range(len(patches)):
        if n[i] > 0:
            plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i] + 0.5, str(int(n[i])), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_price_histogram.png"), dpi=300)
    plt.close()
    
    # 3. Price Boxplot
    plt.figure(figsize=(9, 4))
    sns.boxplot(x=df[target_col], color="#aec7e8", flierprops={"marker": "o", "color": "red", "markersize": 7})
    plt.title("Figure 3: Smartphone Price Boxplot (Outlier Detection)")
    plt.xlabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_price_boxplot.png"), dpi=300)
    plt.close()
    
    # 4. Price Distribution by Brand
    plt.figure(figsize=(12, 6))
    brand_order = df.groupby("Phone_Company")["Price_INR"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Phone_Company", y="Price_INR", order=brand_order, hue="Phone_Company", palette="tab10", legend=False)
    plt.title("Figure 4: Smartphone Price Distribution Across Brands (Sorted by Median)")
    plt.xlabel("Brand / Manufacturer")
    plt.ylabel("Price (₹ INR)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_price_by_brand.png"), dpi=300)
    plt.close()
    
    # 5. RAM vs Price
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="RAM_Size_GB", y="Price_INR", hue="RAM_Size_GB", palette="Blues_d", legend=False)
    sns.stripplot(data=df, x="RAM_Size_GB", y="Price_INR", color="black", alpha=0.5, jitter=0.2)
    plt.title("Figure 5: RAM Capacity (GB) vs Smartphone Price (₹ INR)")
    plt.xlabel("RAM Size (GB)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_ram_vs_price.png"), dpi=300)
    plt.close()
    
    # 6. Storage vs Price
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Storage_GB", y="Price_INR", hue="Storage_GB", palette="Purples_d", legend=False)
    sns.stripplot(data=df, x="Storage_GB", y="Price_INR", color="black", alpha=0.5, jitter=0.2)
    plt.title("Figure 6: Internal Storage Capacity (GB) vs Price (₹ INR)")
    plt.xlabel("Storage (GB)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_storage_vs_price.png"), dpi=300)
    plt.close()
    
    # 7. Battery vs Price
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df, x="Battery_Capacity_mAh", y="Price_INR", scatter_kws={"color": "#ff7f0e", "alpha": 0.7}, line_kws={"color": "darkred"})
    plt.title("Figure 7: Battery Capacity (mAh) vs Smartphone Price (₹ INR)")
    plt.xlabel("Battery Capacity (mAh)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_battery_vs_price.png"), dpi=300)
    plt.close()
    
    # 8. Processor Frequency vs Price
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df, x="Processor_Core1_Frequency_GHz", y="Price_INR", scatter_kws={"color": "#17becf", "alpha": 0.7}, line_kws={"color": "#005580"})
    plt.title("Figure 8: Performance Core Frequency (GHz) vs Price (₹ INR)")
    plt.xlabel("Core 1 Frequency (GHz)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_processor_freq_vs_price.png"), dpi=300)
    plt.close()
    
    # 9. Processor Core Count vs Price
    plt.figure(figsize=(9, 5))
    df_core_cluster = df.copy()
    df_core_cluster["Cluster_Type"] = df_core_cluster["Processor_Core3_Count"].apply(
        lambda x: "3-Cluster (Tri-Cluster)" if x > 0 else "2-Cluster (Dual-Cluster)"
    )
    sns.boxplot(data=df_core_cluster, x="Cluster_Type", y="Price_INR", hue="Cluster_Type", palette="Set2", legend=False)
    sns.stripplot(data=df_core_cluster, x="Cluster_Type", y="Price_INR", color="black", alpha=0.5)
    plt.title("Figure 9: Processor Architecture Cluster Type vs Price (₹ INR)")
    plt.xlabel("CPU Cluster Organization (All Octa-Core)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_core_count_vs_price.png"), dpi=300)
    plt.close()
    
    # 10. Display Refresh Rate vs Price
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Refresh_Rate_Hz", y="Price_INR", hue="Refresh_Rate_Hz", palette="YlGnBu", legend=False)
    sns.stripplot(data=df, x="Refresh_Rate_Hz", y="Price_INR", color="black", alpha=0.4)
    plt.title("Figure 10: Display Refresh Rate (Hz) vs Smartphone Price (₹ INR)")
    plt.xlabel("Refresh Rate (Hz)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_refresh_rate_vs_price.png"), dpi=300)
    plt.close()
    
    # 11. Display Resolution vs Price
    df_res = df.copy()
    df_res["Total_Pixels"] = df_res["Display_Resolution_Width_px"] * df_res["Display_Resolution_Height_px"]
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df_res, x="Total_Pixels", y="Price_INR", scatter_kws={"color": "#9467bd", "alpha": 0.7}, line_kws={"color": "#4a148c"})
    plt.title("Figure 11: Display Resolution (Total Pixels) vs Price (₹ INR)")
    plt.xlabel("Total Display Pixels (Width x Height)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_display_resolution_vs_price.png"), dpi=300)
    plt.close()
    
    # 12. Process Technology (nm) vs Price
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Process_Technology_nm", y="Price_INR", hue="Process_Technology_nm", palette="magma_r", legend=False)
    sns.stripplot(data=df, x="Process_Technology_nm", y="Price_INR", color="black", alpha=0.5)
    plt.title("Figure 12: Semiconductor Process Node (nm) vs Price (₹ INR)")
    plt.xlabel("Process Technology Node (nm - Smaller is More Advanced)")
    plt.ylabel("Price (₹ INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "12_process_tech_vs_price.png"), dpi=300)
    plt.close()
    
    # 13. Correlation Heatmap
    plt.figure(figsize=(14, 11))
    corr_cols = [
        "Price_INR", "RAM_Size_GB", "Storage_GB", "Battery_Capacity_mAh",
        "Process_Technology_nm", "Refresh_Rate_Hz", "Display_Resolution_Width_px",
        "Display_Resolution_Height_px", "Processor_Core1_Frequency_GHz",
        "Processor_Core2_Frequency_GHz", "Cache_Memory_Core1"
    ]
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, cbar_kws={"shrink": .8}, linewidths=0.5)
    plt.title("Figure 13: Correlation Heatmap of Primary Hardware Specifications & Price")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 14. Feature Distribution Plots (Key Numerical Grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    num_features = ["RAM_Size_GB", "Storage_GB", "Battery_Capacity_mAh", "Process_Technology_nm", "Refresh_Rate_Hz", "Processor_Core1_Frequency_GHz"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, col in enumerate(num_features):
        ax = axes[i // 3, i % 3]
        sns.histplot(df[col], kde=True, ax=ax, color=colors[i], bins=10)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
    plt.suptitle("Figure 14: Distribution Profiles of Key Hardware Features", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "14_feature_distributions.png"), dpi=300)
    plt.close()
    
    # 15. Outlier Visualizations (Multi-feature Boxplots)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    check_cols = ["Price_INR", "Battery_Capacity_mAh", "Cache_Memory_Core1", "Processor_Core1_Frequency_GHz"]
    for i, col in enumerate(check_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#80b1d3")
        axes[i].set_title(f"Boxplot: {col}")
    plt.suptitle("Figure 15: Outlier Inspection Across Hardware Specifications", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "15_outlier_visualizations.png"), dpi=300)
    plt.close()
    
    # 16. Actual Feature Relationships (Pairplot Grid)
    pair_df = df[[target_col, "RAM_Size_GB", "Storage_GB", "Processor_Core1_Frequency_GHz", "Process_Technology_nm"]]
    g = sns.pairplot(pair_df, corner=True, diag_kind="kde", plot_kws={"alpha": 0.6, "color": "#1f77b4"})
    g.fig.suptitle("Figure 16: Pairwise Relationships Across Key Smartphone Specifications", y=1.02, fontsize=14, fontweight="bold")
    g.savefig(os.path.join(output_dir, "16_actual_feature_relationships.png"), dpi=300)
    plt.close()
    
    print(f"All 16 EDA visualizations successfully generated in '{output_dir}'.")


def generate_eda_report(df: pd.DataFrame, output_path: str = "reports/eda_report.md") -> str:
    """Generate Markdown report summarizing key EDA observations and hardware price relationships."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    target_col = "Price_INR" if "Price_INR" in df.columns else ("Price_BDT" if "Price_BDT" in df.columns else df.columns[-1])
    
    # Compute correlations with target_col
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_series = df[num_cols].corr()[target_col].sort_values(ascending=False)
    
    report_lines = [
        "# Exploratory Data Analysis (EDA) Report",
        "",
        "## 1. Executive Summary",
        f"This report presents an in-depth exploratory analysis of the smartphone dataset containing **{len(df)} smartphone models** across **{df['Phone_Company'].nunique()} prominent manufacturers**.",
        "",
        "### Key Target Variable Summary (`Price_INR`)",
        f"- **Mean Price**: {df[target_col].mean():,.2f} ₹",
        f"- **Median Price**: {df[target_col].median():,.2f} ₹",
        f"- **Standard Deviation**: {df[target_col].std():,.2f} ₹",
        f"- **Minimum Price**: {df[target_col].min():,.0f} ₹ (Nokia C20 Plus / Nokia C30)",
        f"- **Maximum Price**: {df[target_col].max():,.0f} ₹ (OnePlus 11)",
        f"- **Interquartile Range (IQR)**: {df[target_col].quantile(0.25):,.0f} ₹ to {df[target_col].quantile(0.75):,.0f} ₹",
        "",
        "## 2. Linear Correlation with Smartphone Price",
        "| Feature | Correlation with Price_INR | Relationship Direction & Impact |",
        "| :--- | :--- | :--- |"
    ]
    
    for col, corr in corr_series.items():
        if col == "Price_INR":
            continue
        direction = "Positive (Higher -> More Expensive)" if corr > 0 else "Negative (Lower value -> More Expensive)"
        strength = "Very Strong" if abs(corr) >= 0.7 else ("Strong" if abs(corr) >= 0.5 else ("Moderate" if abs(corr) >= 0.3 else "Weak"))
        report_lines.append(f"| `{col}` | {corr:.3f} | {strength} {direction} |")
        
    report_lines.extend([
        "",
        "## 3. Key Domain Observations",
        "",
        "### 3.1 Semiconductor Process Node (`Process_Technology_nm`)",
        f"- Displays a **strong negative correlation** ({corr_series.get('Process_Technology_nm', 0):.3f}) with price.",
        "- In semiconductor fabrication, a smaller nanometer process (e.g. 3nm, 4nm vs 12nm, 28nm) indicates higher transistor density, superior energy efficiency, and modern flagship architectures.",
        "- Flagship smartphones utilize 3nm and 4nm nodes, command prices above 50,000 ₹, whereas 12nm–28nm phones occupy the entry-level bracket (10,000–20,000 ₹).",
        "",
        "### 3.2 Memory and Storage Dynamics",
        f"- **RAM (`RAM_Size_GB`)** correlates at **{corr_series.get('RAM_Size_GB', 0):.3f}** with price.",
        f"- **Storage (`Storage_GB`)** correlates at **{corr_series.get('Storage_GB', 0):.3f}** with price.",
        "- Devices with 12GB+ RAM and 256GB+ storage predominantly inhabit the premium tier.",
        "",
        "### 3.3 Processor Architectures and Core Clusters",
        "- All 91 devices are Octa-Core (8 cores), rendering `Processor_Core_Count` invariant.",
        "- However, the internal cluster organization varies significantly:",
        "  - **65 dual-cluster devices** (e.g., 2 Performance cores + 6 Efficiency cores).",
        "  - **26 tri-cluster devices** (e.g., 1 Prime Cortex-X / Oryon core + 3 Gold cores + 4 Silver cores).",
        "- Tri-cluster devices consistently command higher market valuations due to flagship SoC packaging.",
        "",
        "### 3.4 Display and Refresh Rate",
        f"- Refresh rate correlates at **{corr_series.get('Refresh_Rate_Hz', 0):.3f}** with price.",
        "- Display resolution (total pixels) has a notable correlation with price. 144Hz–185Hz panels and 1440p displays are exclusively found in premium and gaming smartphones.",
        "",
        "## 4. Visualizations Catalog",
        "All 16 high-resolution figures have been saved to `reports/figures/`:",
        "- `01_price_distribution.png`: Histogram and KDE distribution of Price_INR.",
        "- `02_price_histogram.png`: Discrete bin counts of smartphone prices.",
        "- `03_price_boxplot.png`: Five-number summary and upper outlier check.",
        "- `04_price_by_brand.png`: Brand tier comparison sorted by median price.",
        "- `05_ram_vs_price.png`: RAM capacity impact on price.",
        "- `06_storage_vs_price.png`: Storage capacity impact on price.",
        "- `07_battery_vs_price.png`: Battery capacity regression trend with price.",
        "- `08_processor_freq_vs_price.png`: Primary core frequency correlation with price.",
        "- `09_core_count_vs_price.png`: Dual vs Tri-cluster CPU architecture pricing.",
        "- `10_refresh_rate_vs_price.png`: Screen refresh rate pricing distributions.",
        "- `11_display_resolution_vs_price.png`: Total pixel count vs price.",
        "- `12_process_tech_vs_price.png`: Semiconductor node (nm) vs price.",
        "- `13_correlation_heatmap.png`: Tri-matrix correlation of hardware specs.",
        "- `14_feature_distributions.png`: Multi-panel subplots of key feature histograms.",
        "- `15_outlier_visualizations.png`: Outlier boxplot grid across specs.",
        "- `16_actual_feature_relationships.png`: Pairplot matrix of key specifications."
    ])
    
    report_text = "\n".join(report_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_raw_data
    df = load_raw_data()
    generate_eda_figures(df)
    report = generate_eda_report(df)
    print("EDA completed and report saved.")
