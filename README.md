# Flight Price Prediction Model

## Project Overview

This project aims to develop a machine learning model to accurately predict flight ticket prices based on historical flight data. The dynamic nature of flight prices, influenced by factors such as travel dates, routes, airlines, and number of stops, creates uncertainty for passengers and opportunities for airlines and travel agencies to optimize strategies. This model provides predictions that can assist passengers in making cost-effective travel decisions and help the travel industry refine pricing and recommendations.

## Problem Statement and Background

The variability in flight ticket prices poses challenges for travelers planning their expenses. Airlines, on the other hand, leverage this variability for optimizing revenue. This project focuses on building a predictive model using historical data to forecast flight prices, providing potential benefits for passengers, airlines, and travel agencies.

## Data Introduction

The dataset used in this project was sourced from a public repository on Kaggle. It contains historical flight records with features such as:
* Airline
* Source and Destination cities
* Departure and Arrival times
* Flight duration
* Number of stops
* Ticket prices

### Ethical and Privacy Considerations

The dataset does not include personally identifiable information. However, potential biases may exist, such as an overrepresentation of dominant airlines (Jet Airways, IndiGo) and limited coverage of rare airlines or routes, which could skew predictions. The absence of seat class information also limits the model's ability to differentiate prices based on service quality. Data preprocessing and analysis were conducted with these potential biases in mind.

## Methodology

To achieve accurate price predictions, the following data science techniques were employed:

1.  **Data Preprocessing:**
    * Extracted meaningful date and time features ('Journey\_Day', 'Journey\_Month', 'Dep\_Hour', 'Dep\_Minute', 'Arrival\_Hour', 'Arrival\_Minute') from 'Date\_of\_Journey', 'Dep\_Time', and 'Arrival\_Time'.
    * Converted 'Duration' to 'Duration\_Minutes' for numerical processing.
    * Handled missing values in 'Total\_Stops' by filling with 'non-stop' and mapped stop descriptions to numerical values.
    * Dropped unnecessary columns such as 'Date\_of\_Journey', 'Dep\_Time', 'Arrival\_Time', 'Route', and 'Additional\_Info'.
    * Encoded categorical features ('Airline', 'Source', 'Destination') using Label Encoding.
    * Standardized numerical features for uniform scaling.

2.  **Exploratory Data Analysis (EDA):**
    * Generated a correlation matrix to identify relationships between numerical features, observing a strong positive correlation between 'Duration\_Minutes', 'Total\_Stops', and 'Price'.
    * Visualized the percentage distribution of airlines.
    * Visualized price distribution by the number of stops using a boxplot, showing that non-stop flights tend to have lower and more consistent prices.

3.  **Model Selection and Training:**
    * Used Linear Regression as a baseline model (RMSE: 3384.22, R²: 0.45).
    * Implemented K-Nearest Neighbors (KNN) Regressor.
    * Optimized KNN hyperparameters ('n\_neighbors', 'weights', 'metric') using GridSearchCV with 5-fold cross-validation.
    * Evaluated the tuned KNN model on a validation set.

## Results and Conclusions

The tuned KNN model demonstrated strong performance, achieving an RMSE of 2138.79 and an R² score of 0.78 (based on the report, the notebook shows R²: 0.76).

Key influential predictors identified were:
* **Total Stops:** A higher number of stops was generally associated with higher prices and greater variability.
* **Duration:** Longer flights tended to have higher prices.
* **Journey Month:** Seasonal trends, particularly peak travel months, influenced average prices.

Challenges observed included the model underpredicting high-priced flights due to their rarity and potential bias from data imbalance related to airline and route representation.

The actual vs. predicted values plot showed that most predictions were close to the actual values, while outliers were harder to predict. The residual plot suggested that the model fit the data well for the most part, with residuals randomly scattered around zero.

## Future Work

Potential enhancements to the project include:
* Implementing advanced regression models such as Gradient Boosting methods (XGBoost, LightGBM) to potentially improve accuracy.
* Incorporating additional relevant features like seat class, holiday indicators, or real-time demand data.
* Applying time series analysis techniques (ARIMA, Prophet) to study how prices change dynamically as the departure date approaches.
* Addressing data imbalance by expanding the dataset to include a more balanced representation of airlines and routes.

## How to Run the Code

1.  Clone this repository.
2.  Install the required libraries using `pip install -r requirements.txt`.
3.  Ensure the `Data_Train.csv` and `Test_set.csv` files are in the same directory as the notebook.
4.  Open and run the `flight_price_prediction_model.ipynb` notebook in a Jupyter environment.

## Contact

Akshat Vij
Email: vij.a@northeastern.edu
