# Gold Price Prediction using Random Forest Regressor

## Project Overview
This project leverages machine learning to predict the price of Gold (GLD) based on a dataset containing historical financial indicators. Utilizing a Random Forest Regressor, the model aims to provide accurate predictions, achieving a high R-squared score, and demonstrating strong performance in tracking actual gold price movements.

## Dataset
The analysis is performed on the `gld_price_data.csv` dataset, which comprises the following key features:
- **`Date`**: The date of the recorded observation.
- **`SPX`**: The S&P 500 Index, representing a broad market indicator.
- **`GLD`**: The target variable, representing the Gold Price (USD).
- **`USO`**: The United States Oil Fund ETF, reflecting oil market trends.
- **`SLV`**: The iShares Silver Trust ETF, indicating silver prices.
- **`EUR/USD`**: The Euro to US Dollar exchange rate, a currency market indicator.

## Technologies Used
This project is developed using Python and relies on the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn` (sklearn)

## Getting Started
To run this project, follow these steps:

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Ensure you have the dataset**: Place the `gld_price_data.csv` file in the same directory as the Jupyter/Colab notebook, or update the path in the code.
3.  **Install dependencies**: Make sure you have all the required Python libraries installed. You can install them using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Open and run the notebook**: Open `gold_price_prediction.ipynb` in a Jupyter environment or Google Colab and execute the cells sequentially.

## Methodology
The project systematically approaches the prediction task through the following stages:

### 1. Data Loading and Initial Exploration
- The `gld_price_data.csv` is loaded into a pandas DataFrame.
- Initial data inspection includes `head()`, `tail()`, `shape`, `info()`, and `describe()` to understand its structure and statistical properties.
- Missing values are identified using `isnull().sum()`.
- The 'Date' column is converted to a datetime object for proper time-series analysis capabilities.

### 2. Exploratory Data Analysis (EDA)
- **Correlation Matrix**: A heatmap visualization of the correlation matrix is generated using `seaborn` to illustrate relationships between features and the target variable (`GLD`).
- **Target Variable Distribution**: The distribution of the `GLD` price is visualized using a `distplot` to understand its spread and characteristics.

### 3. Data Preprocessing
- **Feature Engineering**: The 'Date' column is dropped from the features (`X`) as it's not directly used in the model training after being used for correlation analysis, and `GLD` is separated as the target variable (`y`).
- **Train-Test Split**: The dataset is divided into training (80%) and testing (20%) sets using `train_test_split` with `random_state=2` for reproducibility.

### 4. Model Training
- A `RandomForestRegressor` model is initialized with `n_estimators=100`.
- The model is trained on the prepared training data (`x_train`, `y_train`).

### 5. Model Evaluation and Visualization
- **Prediction**: The trained model generates predictions on the unseen test set (`x_test`).
- **R-squared Score**: The model's performance is quantified using the R-squared metric, comparing predicted (`test_data_prediction`) against actual (`y_test`) values.
- **Visual Comparison**: A plot is generated to visually compare the actual GLD prices with the predicted GLD prices, showcasing the model's ability to capture trends.

## Results
The Random Forest Regressor model achieved an impressive R-squared score of **0.9891** on the test set. This indicates a very strong predictive capability, with the model explaining approximately 98.91% of the variance in gold prices. The visual comparison between actual and predicted prices further confirms the model's high accuracy and close tracking of gold price movements.

## Future Enhancements
- **Hyperparameter Tuning**: Optimize the `RandomForestRegressor` parameters (e.g., `max_depth`, `min_samples_split`) for potentially better performance.
- **Time Series Cross-Validation**: Implement more robust validation strategies tailored for time-series data.
- **Additional Features**: Explore incorporating other financial indicators, macroeconomic data, or news sentiment as features.
- **Deep Learning Models**: Experiment with advanced models like LSTMs or Transformers for time-series forecasting.
- **Deployment**: Develop an API or a simple web application to deploy the trained model for real-time predictions.
