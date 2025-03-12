# Medical Insurance Cost Prediction

## Overview

This project predicts medical insurance costs based on demographic and health-related features using machine learning techniques. The model is trained using Linear Regression and evaluates different statistical metrics to measure its performance.

## Business Problem

Medical insurance companies need a reliable way to estimate the cost of insurance based on various factors such as age, BMI, smoking habits, and region. Predicting these costs accurately allows insurance providers to optimize pricing strategies, manage risk, and offer fair premiums to customers. This project aims to provide a data-driven approach to estimating insurance charges, reducing uncertainty and improving decision-making for both insurers and policyholders.

## Objectives

- Develop a machine learning model to predict medical insurance costs.
- Analyze the impact of different factors such as age, BMI, smoking habits, and region on insurance costs.
- Implement a user-friendly web interface for real-time cost prediction.
- Evaluate the model using statistical metrics such as R-squared and accuracy scores.
- Save and load the trained model using Pickle for easy deployment.

## Features

- Web interface using Flask, HTML, and CSS for user-friendly predictions.
- Data preprocessing, including encoding categorical variables.
- Exploratory Data Analysis (EDA) for understanding data distribution.
- Training and evaluating a Linear Regression model.
- Model evaluation using R-squared and accuracy scores.
- Building a predictive system for real-time cost estimation.
- Saving and loading the trained model using Pickle.

## Technologies Used

- Flask (for web deployment)
- HTML, CSS (for front-end design)
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Pickle (for model persistence)

## Setup Instructions

1. Download the ZIP file and extract it.
2. Navigate to the extracted folder.
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open the web interface and enter input details:
   ```
   http://127.0.0.1:5000/
   ```
5. The trained model will be saved as `regmodel.pkl`.

## Dataset

- The dataset used contains the following features:
  - **Age**: Age of the insured person
  - **Sex**: Gender (Male/Female)
  - **BMI**: Body Mass Index
  - **Children**: Number of dependents
  - **Smoker**: Whether the person is a smoker (Yes/No)
  - **Region**: Geographic location
  - **Charges**: Medical insurance cost (target variable)

## Model Training

- **Linear Regression** is used for predicting insurance costs.
- The dataset is split into **training (80%)** and **testing (20%)** data.
- Model performance is evaluated using:
  - R-squared score on training and testing data.
  - Accuracy score of the regression model.

## Predicting Insurance Cost

To predict medical insurance costs for a new input:

1. Define input values:
   ```python
   input_data = (31,1,25.74,0,1,0)  # Example input
   ```
2. Convert input data to a NumPy array:
   ```python
   input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
   ```
3. Load the trained model and make predictions:
   ```python
   pickled_model = pickle.load(open('regmodel.pkl', 'rb'))
   prediction = pickled_model.predict(input_data_as_numpy_array)
   print('The insurance cost is USD', prediction[0])
   ```

## Future Enhancements

- Deploy the model as a web application using Flask.
- Integrate real-world medical insurance data for better predictions.

## Contributing

Feel free to submit issues and pull requests if you want to contribute to this project.

## License

This project is licensed under the MIT License.

