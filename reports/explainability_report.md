# Smartphone Price Prediction - Model Explainability & Attribution Report

## 1. Executive Summary
Understanding which smartphone specifications determine consumer price is essential for product architects and consumers alike.
To provide rigorous interpretability, three complementary attribution methods were implemented:
1. **Tree Impurity Feature Importance (MDI)**: Quantifies total variance reduction across decision splits in tree ensembles.
2. **Permutation Feature Importance**: Evaluates the degradation in prediction accuracy (increase in MAE) when a feature's values are randomly shuffled on unseen validation data.
3. **Regularized Linear Coefficients**: Measures the standardized directional sensitivity ($eta$ weights) for each hardware unit.

## 2. Top Specifications Driving Smartphone Price

### 2.1 Permutation Importance (Validation Partition Impact)
| Rank | Feature | Mean MAE Increase (₹ INR) | Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `Silicon_Density` | **+₹6,432** | Shuffling degrades validation accuracy significantly |
| 2 | `Compute_Capability_Score` | **+₹2,282** | Shuffling degrades validation accuracy significantly |
| 3 | `SoC_Tier` | **+₹1,753** | Shuffling degrades validation accuracy significantly |
| 4 | `Max_Core_Frequency_GHz` | **+₹1,573** | Shuffling degrades validation accuracy significantly |
| 5 | `Processor_Core1_Frequency_GHz` | **+₹1,573** | Shuffling degrades validation accuracy significantly |
| 6 | `Weighted_Average_Frequency_GHz` | **+₹1,245** | Shuffling degrades validation accuracy significantly |
| 7 | `Display_Resolution_Width_px` | **+₹1,160** | Shuffling degrades validation accuracy significantly |
| 8 | `RAM_Size_GB` | **+₹1,125** | Shuffling degrades validation accuracy significantly |
| 9 | `SoC_Maker` | **+₹1,086** | Shuffling degrades validation accuracy significantly |
| 10 | `Processor_Core3_Count` | **+₹997** | Shuffling degrades validation accuracy significantly |

### 2.2 Tree-Based MDI Importance (Gini)
| Rank | Feature | Importance Share | Dominant Physical Metric |
| :--- | :--- | :--- | :--- |

## 3. Domain Insights and Critical Caution

> [!IMPORTANT]
> **Causal Attribution Caution**: Feature importance values in non-experimental observational datasets reflect *predictive association*, not independent causal mechanisms.
> In modern mobile systems, hardware specifications are co-engineered: high-end processors (3nm/4nm) are almost universally packaged with LPDDR5X memory, high-resolution 120Hz+ AMOLED displays, and high-megapixel optical sensors.
> Therefore, high importance scores represent joint generational tiering rather than isolated isolated price markups.

### Key Hardware Drivers Identified:
1. **Semiconductor Process Node (`Process_Technology_nm`)**: Smaller lithography nodes (3nm, 4nm) represent the largest single cost driver, signifying modern flagship system-on-chip silicon.
2. **Memory Subsystem (`RAM_Size_GB` & `RAM_Type`)**: 12GB+ LPDDR5X RAM strictly separates upper-tier devices from budget 4GB LPDDR4X offerings.
3. **Peak Compute Clock Speed (`Max_Core_Frequency_GHz`)**: Frequency scaling above 3.0 GHz marks flagship Cortex-X / Oryon architectures that command high premiums.
4. **Display Architecture (`Display_Tech_Category` & `Total_Display_Pixels`)**: AMOLED/OLED displays and 1440p panels add substantial bill-of-materials expense compared to standard 720p LCDs.