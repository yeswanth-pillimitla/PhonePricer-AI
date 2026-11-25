# Feature Selection and Dimensionality Audit Report

## 1. Overview
The dataset provides 30 columns (29 potential features + 1 target: `Price_INR`).
Following domain-driven feature engineering and univariate F-regression filtering, a total of **21 numerical features** and **6 categorical features** (27 total input predictors) were selected.

## 2. Selected Feature Matrix

### 2.1 Numerical Features (21 Predictors)
| Feature Name | Type | Physical Domain Interpretation |
| :--- | :--- | :--- |
| `RAM_Size_GB` | int64 | Base system memory (primary performance driver) |
| `Storage_GB` | int64 | Base internal flash storage capacity |
| `Battery_Capacity_mAh` | int64 | Battery energy storage capacity in milliampere-hours |
| `Process_Technology_nm` | int64 | Semiconductor lithography node (lower = superior density/efficiency) |
| `Display_Resolution_Width_px` | int64 | Panel horizontal pixel density |
| `Display_Resolution_Height_px` | int64 | Panel vertical pixel density |
| `Refresh_Rate_Hz` | int64 | Display refresh capability (60Hz up to 185Hz) |
| `Processor_Core1_Frequency_GHz` | float64 | Clock speed of prime performance core cluster |
| `Processor_Core2_Frequency_GHz` | float64 | Clock speed of secondary performance/balanced cluster |
| `Processor_Core3_Count` | int64 | Active efficiency/little core count in cluster 3 |
| `Processor_Core3_Frequency_GHz` | float64 | Clock speed of efficiency cluster |
| `Cache_Memory_Core3` | int64 | On-die L2/L3 cache allocated to efficiency cluster |
| `Max_Core_Frequency_GHz` | float64 | Peak single-core burst frequency |
| `Weighted_Average_Frequency_GHz` | float64 | Compute throughput across all cores |
| `Total_Display_Pixels` | int64 | Total active display resolution (Width x Height) |
| `Display_Bandwidth_MP_s` | float64 | Pixel throughput per second (Pixels x Refresh Rate) |
| `Silicon_Density` | float64 | Clock speed relative to lithography node (GHz / nm) |
| `Total_Cache_KB` | int64 | Aggregate processor cache across all clusters |
| `Num_CPU_Clusters` | int64 | Architecture topology indicator (2 clusters vs 3 clusters) |
| `Memory_Score` | int64 | Memory composite capacity (RAM x Storage) |
| `Compute_Capability_Score` | float64 | High-end compute index (RAM x Max_Freq / nm) |

### 2.2 Categorical Features (6 Predictors)
| Categorical Feature | Cardinality | Justification |
| :--- | :--- | :--- |
| `Phone_Company` | 11 | Brand premium, build quality reputation, customer segment |
| `RAM_Type` | 4 | Memory generation (LPDDR4X vs LPDDR5X bandwidth) |
| `ISA` | 2 | Instruction set architecture tier (ARMv8 vs ARMv9) |
| `Display_Tech_Category` | 2 | Display panel hardware tier (AMOLED/OLED vs LCD) |
| `SoC_Tier` | 4 | Market silicon performance tier (Flagship, Premium Mid, Mid, Budget) |
| `SoC_Maker` | 5 | Silicon design company (Qualcomm, MediaTek, Exynos, Unisoc, Other) |

## 3. Discarded Features Justification

| Discarded Feature | Reason for Exclusion |
| :--- | :--- |
| `Phone_Model` | High cardinality (91 distinct values for 91 rows). Causes extreme overfitting and zero generalization. |
| `SoC` | High raw cardinality (57 unique names). Summarized cleanly into `SoC_Tier` and `SoC_Maker`. |
| `Processor_Core_Count` | Zero variance (all 91 smartphones in this dataset have exactly 8 cores). Constant features carry zero information. |
| `Display_Aspect_Ratio` | Negligible variance (majority clustered at 20:9 or 20.5:9). Low predictive value. |
| `Rear_Camera_MP` | 50MP is universally present across 12,000 BDT and 70,000 BDT phones, acting as noise for regression. |
| `Processor_Core1_Count`, `Processor_Core2_Count` | Constant or near-constant cluster splits with minimal independent pricing signal compared to frequencies. |
| `Cache_Memory_Core1`, `Cache_Memory_Core2` | Redundant with `Total_Cache_KB` and missing in several SoC variants. |
| `Frequency_of_Different_Cores` | Redundant with individual core frequency columns. |
| `Display_Resolution` | Redundant string with width and height pixel columns. |
| `Camera_Specifications` | Unstructured text string with high noise; parsed into numerical and category equivalents. |
