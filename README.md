# House Prices — Categorical Encoding Techniques

This project explores various **categorical encoding methods** applied to the popular Kaggle [House Prices: Advanced Regression Techniques](https://kaggle.com) dataset. 

The purpose of this notebook is to systematically evaluate how different encoding strategies impact the Root Mean Squared Error (RMSE) of a regularized baseline validation layout using a linear model.

---

## 📋 Project Structure

The project cleans, splits, processes, and validates features over a sequence of progressive methodologies:
1. **Target Feature Engineering:** Predicting \(\log(1 + \text{SalePrice})\) instead of the raw values to balance errors caused by extreme outliers/skewed distributions.
2. **Missing Value Imputation:** Filling numeric features with median values and categorical arrays with a placeholder string (`'Missing'`) to handle missing indicators securely.
3. **Encoding Evaluation Baseline:** Training an isolated linear model using exclusively numeric properties.
4. **Encoding Implementations:** Iterating through categorical adjustments, tracking comparative errors, and mapping results.

---

## 🛠️ Encoding Techniques Implemented

- **Baseline:** Ignores categorical elements entirely and trains solely on numeric columns.
- **Label Encoding:** Substitutes string classifications with nominal indicators (\(0, 1, 2, \dots\)) using scikit-learn's `LabelEncoder`.
- **One-Hot Encoding:** Generates binary column vectors for every label present (via pandas `get_dummies`), ignoring structural relationships while increasing dimensionality.
- **Feature Hashing:** Compresses vast string designations into a predefined, static array size using a structural hashing model (`FeatureHasher`).
- **Dataset Statistics Encoding (Frequency Encoding):** Replaces categories with their explicit total occurrence frequency count within the training frame.
- **Cyclic Encoding:** Converts variables with chronological circular behaviors (like month sold, `MoSold`) using sine and cosine functions to accurately capture chronological distance.
- **Target Encoding (Leaky / Out-of-Fold Global):** Recodes feature properties using the conditional mean of the destination value.
- **K-Fold Target Encoding:** Prevents global data leakage and overfitting by calculating conditional category properties strictly using out-of-fold partitions across a 5-fold cross-validation scheme.

---

## 📊 Experimental Results

Below is the execution log tracking validation performance across models, sorted from lowest error (best) to highest:

| Rank | Encoding Strategy | Validation RMSE |
| :---: | :--- | :---: |
| 1 | Target Encoding (Leaky)* | 0.135139 |
| 2 | **K-Fold Target Encoding** | **0.139861** |
| 3 | Dataset Statistics + Cyclic | 0.144420 |
| 4 | Dataset Statistics Encoding | 0.144813 |
| 5 | Feature Hashing | 0.146441 |
| 6 | Baseline (Numeric Only) | 0.151864 |
| 7 | Label Encoding | 0.155414 |
| 8 | One-Hot Encoding | 0.173683 |

*\*Note: The leaky target encoding score represents a structural data leak (the row's own price is included in its historical calculation group), leading to artificial score inflation. Out-of-Fold K-Fold Target Encoding provides the best valid score.*

---

## 🔑 Key Takeaways

- **High Cardinality Constraints:** One-Hot Encoding degraded the performance of the linear model. Expanding properties like `Neighborhood` into dozens of independent indicator columns diluted structural signal strength.
- **Target Encoding Efficiency:** Target values mapped dynamically over categories provided structural boosts, lowering test error once properly guarded inside an **Out-of-Fold K-Fold structure**.
- **Linear Model Limits with Ordered Assumptions:** Label Encoding introduced artificial continuous sequence lines (0 < 1 < 2), misleading the Linear Regression engine into looking for progressive trends that did not exist.

---

## 💻 Technical Requirements

Ensure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib
```

To run this experiment, place your Kaggle data files (`train.csv`, `test.csv`) in the root directory alongside the workbook file, then launch and execute your Jupyter instance.
