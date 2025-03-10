# Rainfall Prediction Project

## Overview

This project focuses on predicting rainfall using machine learning techniques. The dataset includes various meteorological parameters that influence rainfall patterns. The model is trained to provide accurate rainfall predictions based on historical weather data.

## Features

- Data preprocessing, including handling missing values and outlier removal
- Exploratory Data Analysis (EDA) to understand data distribution
- Balancing the dataset to improve model performance
- Training and evaluating machine learning models
- Flask-based API for serving predictions
- Web interface using HTML
- Model file for offline predictions

## Technologies Used

- Python
- Flask (for API development)
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- HTML/CSS (for web interface)

## Setup Instructions

1. Download the ZIP file and extract it.
2. Navigate to the extracted folder.
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open the web interface and make your predictions
   ```
   http://127.0.0.1:5000/
   ```

## Running on Google Colab

1. Upload the dataset to Google Colab.
2. Execute the cells step by step to preprocess data, train the model, and analyze results.

## Running the Flask API

1. Navigate to the directory containing the Flask app.
2. Run the Flask application:
   ```sh
   python app.py
   ```
3. The API will be available at `http://127.0.0.1:5000/`.

## Web Interface

- The project includes an HTML file to interact with the model.
- Open the HTML file in a browser to input values and get predictions.

## Model File

- A pre-trained model file (`.pkl` or `.h5`) is included in the project.
- This file can be used for making offline predictions without retraining.

## Future Enhancements

- Integration with real-time weather APIs for live predictions.
- Exploring deep learning models for improved accuracy.

## Contributing

Feel free to submit issues and pull requests if you want to contribute to this project.

## License

This project is licensed under the MIT License.

