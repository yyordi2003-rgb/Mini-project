# Ride Price Prediction ML Project

## Project Overview
This project focuses on building a machine learning pipeline to predict ride-sharing prices using a synthetic, "real-world messy" dataset. The primary goal was to demonstrate end-to-end data science skills, including **data synthesis**, **rigorous data cleaning** and **regression modeling**. By deliberately injecting noise, outliers, and inconsistencies, this project showcases the ability to handle raw, imperfect data common in production environments.

## Dataset Description
The dataset consists of 150 simulated ride records. To simulate real-world data collection issues, the following "messy" attributes were included:
* **Outliers:** Extreme values in trip duration (e.g., 999 minutes) and pricing to test model robustness.
* **Missing Values:** Approximately 10% null values in the `Traffic_level` feature to practice imputation strategies.
* **Inconsistencies:** Variations in categorical labeling (e.g., "Rainy", "rain", "RAINY") requiring string standardization.
* **Logical Errors:** Negative distances and zero-distance high-cost trips to simulate GPS glitches.



## Features Used 
The following features were utilized to train the model:
* **Distance (km)**
* **Trip duration (minutes)**
* **Time of day**
* **Traffic level**
* **Weather condition**
* **Demand level**
* **Day Type (Weekend vs. Weekday)** — Added to capture different demand patterns between commute and leisure days.

## How to Run the Notebook
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/ride-price-ml.git](https://github.com/your-username/ride-price-ml.git)

## 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
## 3. Execute the Notebook

Open `notebook/ride_price_model.ipynb` in **Jupyter Notebook** or **VS Code** and run all cells sequentially.

---

## Key Findings

### Outlier Impact
The model's RMSE was significantly improved after handling the **999-minute trip duration outlier**, which initially skewed the linear regression coefficients.

### Feature Importance
`Distance_km` and `trip_duration` were the strongest predictors of price.

### Encoding Success
Using a `ColumnTransformer` to apply `OneHotEncoder` and `OrdinalEncoder` appropriately prevented the model from assigning false mathematical rankings to nominal data.

---------
#### Author: Yordanos Teshome
