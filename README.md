# House Prices Regression Analysis

A machine learning project for predicting house prices using regression techniques. This project implements and compares multiple regression models including Linear Regression, Ridge, and Lasso with comprehensive data preprocessing and feature engineering.

## 📊 Project Overview

This project analyzes the Ames Housing dataset to predict house sale prices. The analysis includes:
- Exploratory Data Analysis (EDA)
- Missing value imputation
- Feature engineering and encoding
- Multiple regression model implementations
- Model performance comparison
- Visualization of predictions and residuals

## 📁 Project Structure

```
house_prices_regression/
│
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── sample_submission.csv  # Sample submission format
│
├── 01_data_loading_and_summary.ipynb  # Main analysis notebook
└── README.md                          # Project documentation
```

## 🔍 Dataset

The dataset contains house sale data with 79 explanatory variables describing various aspects of residential homes:
- **Target Variable**: SalePrice
- **Features**: Lot size, quality ratings, year built, number of rooms, basement info, garage details, and more
- **Training Set**: ~1460 samples
- **Test Set**: ~1459 samples

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning models and preprocessing
- **matplotlib** - Data visualization

## 📋 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house_prices_regression.git
cd house_prices_regression
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook 01_data_loading_and_summary.ipynb
```

## 🚀 Methodology

### 1. Data Preprocessing
- **Missing Values Handling**:
  - Categorical features with NA as meaningful category: Filled with "None"
  - Numeric features: Imputed with median values
  - Mode imputation for categorical features

### 2. Feature Engineering
- One-hot encoding for categorical variables
- Log transformation of target variable (SalePrice) to handle skewness
- Feature scaling using StandardScaler

### 3. Models Implemented

#### Linear Regression
- Baseline model for comparison
- Applied with log-transformed target

#### Ridge Regression (L2 Regularization)
- Cross-validated alpha selection
- Tested alphas: [0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
- Pipeline with StandardScaler

#### Lasso Regression (L1 Regularization)
- Automatic feature selection
- Cross-validated alpha selection
- Maximum iterations: 20,000

### 4. Model Evaluation
- **Metrics**: RMSE (Root Mean Squared Error), R² Score
- **Validation**: Train-test split (80-20) with random_state=42
- **Visualization**: 
  - Actual vs Predicted scatter plots
  - Residual plots
  - Residual distribution histograms

## 📈 Results

The project compares three regression models:

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression (Log) | ~$X,XXX | ~0.XX |
| Ridge Regression | ~$X,XXX | ~0.XX |
| Lasso Regression | ~$X,XXX | ~0.XX |

*Note: Run the notebook to see actual results*

## 💡 Key Insights

- Log transformation of the target variable significantly improves model performance
- Regularization techniques (Ridge/Lasso) help prevent overfitting
- Feature encoding increases dimensionality but captures categorical relationships
- Residual analysis shows model behavior and potential areas for improvement

## 🎯 Future Improvements

- [ ] Advanced feature engineering (polynomial features, interactions)
- [ ] Ensemble methods (Random Forest, Gradient Boosting, XGBoost)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Feature importance analysis
- [ ] Cross-validation with multiple folds
- [ ] Outlier detection and handling
- [ ] Additional feature selection techniques

## 📝 Usage

To reproduce the analysis:

1. Open the notebook `01_data_loading_and_summary.ipynb`
2. Run cells sequentially from top to bottom
3. Analyze the outputs and visualizations
4. Experiment with different models or hyperparameters

## 📊 Visualizations

The project includes:
- **Actual vs Predicted**: Scatter plot comparing true and predicted prices
- **Residual Plot**: Shows prediction errors across different price ranges
- **Residual Distribution**: Histogram showing error distribution

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests with improvements
- Share your results with different approaches

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Your Name - [Your GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Inspiration from the Kaggle community

---

**Note**: This is an educational project for learning regression techniques and machine learning workflows.
