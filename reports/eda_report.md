# Exploratory Data Analysis (EDA) Report

## 1. Executive Summary
This report presents an in-depth exploratory analysis of the smartphone dataset containing **91 smartphone models** across **11 prominent manufacturers**.

### Key Target Variable Summary (`Price_INR`)
- **Mean Price**: 26,784.54 ₹
- **Median Price**: 22,651.00 ₹
- **Standard Deviation**: 15,222.44 ₹
- **Minimum Price**: 7,550 ₹ (Nokia C20 Plus / Nokia C30)
- **Maximum Price**: 75,494 ₹ (OnePlus 11)
- **Interquartile Range (IQR)**: 15,289 ₹ to 34,164 ₹

## 2. Linear Correlation with Smartphone Price
| Feature | Correlation with Price_INR | Relationship Direction & Impact |
| :--- | :--- | :--- |
| `Processor_Core2_Frequency_GHz` | 0.793 | Very Strong Positive (Higher -> More Expensive) |
| `Processor_Core1_Frequency_GHz` | 0.782 | Very Strong Positive (Higher -> More Expensive) |
| `Display_Resolution_Width_px` | 0.752 | Very Strong Positive (Higher -> More Expensive) |
| `Display_Resolution_Height_px` | 0.746 | Very Strong Positive (Higher -> More Expensive) |
| `RAM_Size_GB` | 0.724 | Very Strong Positive (Higher -> More Expensive) |
| `Storage_GB` | 0.616 | Strong Positive (Higher -> More Expensive) |
| `Processor_Core3_Frequency_GHz` | 0.573 | Strong Positive (Higher -> More Expensive) |
| `Processor_Core3_Count` | 0.521 | Strong Positive (Higher -> More Expensive) |
| `Cache_Memory_Core1` | 0.510 | Strong Positive (Higher -> More Expensive) |
| `Cache_Memory_Core2` | 0.509 | Strong Positive (Higher -> More Expensive) |
| `Refresh_Rate_Hz` | 0.497 | Moderate Positive (Higher -> More Expensive) |
| `Cache_Memory_Core3` | 0.495 | Moderate Positive (Higher -> More Expensive) |
| `Battery_Capacity_mAh` | 0.309 | Moderate Positive (Higher -> More Expensive) |
| `Processor_Core2_Count` | -0.294 | Weak Negative (Lower value -> More Expensive) |
| `Processor_Core1_Count` | -0.437 | Moderate Negative (Lower value -> More Expensive) |
| `Process_Technology_nm` | -0.465 | Moderate Negative (Lower value -> More Expensive) |
| `Processor_Core_Count` | nan | Weak Negative (Lower value -> More Expensive) |

## 3. Key Domain Observations

### 3.1 Semiconductor Process Node (`Process_Technology_nm`)
- Displays a **strong negative correlation** (-0.465) with price.
- In semiconductor fabrication, a smaller nanometer process (e.g. 3nm, 4nm vs 12nm, 28nm) indicates higher transistor density, superior energy efficiency, and modern flagship architectures.
- Flagship smartphones utilize 3nm and 4nm nodes, command prices above 50,000 ₹, whereas 12nm–28nm phones occupy the entry-level bracket (10,000–20,000 ₹).

### 3.2 Memory and Storage Dynamics
- **RAM (`RAM_Size_GB`)** correlates at **0.724** with price.
- **Storage (`Storage_GB`)** correlates at **0.616** with price.
- Devices with 12GB+ RAM and 256GB+ storage predominantly inhabit the premium tier.

### 3.3 Processor Architectures and Core Clusters
- All 91 devices are Octa-Core (8 cores), rendering `Processor_Core_Count` invariant.
- However, the internal cluster organization varies significantly:
  - **65 dual-cluster devices** (e.g., 2 Performance cores + 6 Efficiency cores).
  - **26 tri-cluster devices** (e.g., 1 Prime Cortex-X / Oryon core + 3 Gold cores + 4 Silver cores).
- Tri-cluster devices consistently command higher market valuations due to flagship SoC packaging.

### 3.4 Display and Refresh Rate
- Refresh rate correlates at **0.497** with price.
- Display resolution (total pixels) has a notable correlation with price. 144Hz–185Hz panels and 1440p displays are exclusively found in premium and gaming smartphones.

## 4. Visualizations Catalog
All 16 high-resolution figures have been saved to `reports/figures/`:
- `01_price_distribution.png`: Histogram and KDE distribution of Price_INR.
- `02_price_histogram.png`: Discrete bin counts of smartphone prices.
- `03_price_boxplot.png`: Five-number summary and upper outlier check.
- `04_price_by_brand.png`: Brand tier comparison sorted by median price.
- `05_ram_vs_price.png`: RAM capacity impact on price.
- `06_storage_vs_price.png`: Storage capacity impact on price.
- `07_battery_vs_price.png`: Battery capacity regression trend with price.
- `08_processor_freq_vs_price.png`: Primary core frequency correlation with price.
- `09_core_count_vs_price.png`: Dual vs Tri-cluster CPU architecture pricing.
- `10_refresh_rate_vs_price.png`: Screen refresh rate pricing distributions.
- `11_display_resolution_vs_price.png`: Total pixel count vs price.
- `12_process_tech_vs_price.png`: Semiconductor node (nm) vs price.
- `13_correlation_heatmap.png`: Tri-matrix correlation of hardware specs.
- `14_feature_distributions.png`: Multi-panel subplots of key feature histograms.
- `15_outlier_visualizations.png`: Outlier boxplot grid across specs.
- `16_actual_feature_relationships.png`: Pairplot matrix of key specifications.