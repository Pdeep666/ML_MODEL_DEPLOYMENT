# Restaurant Sentiment Analysis

## Overview

This project focuses on sentiment analysis for restaurant reviews. It aims to classify customer feedback as positive or negative using machine learning and natural language processing (NLP) techniques. The model is trained using different classifiers and deployed via a Flask-based web interface.

## Business problem

Understanding customer sentiment is crucial for restaurants to improve their services and customer satisfaction. Traditional feedback mechanisms may not provide actionable insights due to the vast number of reviews available. This project aims to automate the sentiment analysis process, enabling restaurants to classify reviews as positive or negative efficiently. The insights gained from this analysis can help restaurants identify strengths and areas needing improvement, ultimately leading to better customer experiences.

## Objectives

- Develop a sentiment analysis model to classify restaurant reviews as positive or negative.
- Preprocess textual data, including cleaning, tokenization, and vectorization.
- Perform Exploratory Data Analysis (EDA) to understand sentiment distribution.
- Train and evaluate multiple machine learning models (Naive Bayes, Logistic Regression, Random Forest).
- Deploy the model using a Flask-based API for real-time sentiment classification.
- Create a web-based interface for user-friendly interaction.
- Save and load the trained model for offline predictions.

## Features

- Data preprocessing, including text cleaning and tokenization
- Exploratory Data Analysis (EDA) with word clouds
- Feature extraction using CountVectorizer
- Training multiple machine learning models (Naive Bayes, Logistic Regression, Random Forest)
- Flask-based API for real-time sentiment analysis
- Web interface using HTML and CSS for user interaction
- Model saving and loading for offline predictions

## Technologies Used

- Python
- Flask (for API development)
- Pandas, NumPy
- Scikit-learn, NLTK
- Matplotlib, Seaborn
- WordCloud (for visualization)
- HTML/CSS (for web interface)

## Setup Instructions

1. Download the ZIP file and extract it.
2. Navigate to the extracted folder.
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open the web interface to analyze sentiments:
   ```
   http://127.0.0.1:5000/
   ```

## Running on Google Colab

1. Upload the dataset (`Restaurant_Reviews.tsv`) to Google Colab.
2. Load and preprocess the data.
3. Train the sentiment analysis models.
4. Evaluate and visualize sentiment trends.
5. Save the trained model (`sentiment_model.pkl`).

## Running the Flask API

1. Navigate to the directory containing the Flask app.
2. Run the Flask application:
   ```sh
   python app.py
   ```
3. The API will be available at `http://127.0.0.1:5000/`.
4. Enter a restaurant review in the web interface to predict its sentiment.

## Web Interface

- The project includes an HTML file for a user-friendly sentiment prediction interface.
- Open the web interface in a browser and input restaurant reviews to get sentiment predictions.
![image_anti](https://github.com/Pdeep666/ML_MODEL_DEPLOYMENT/blob/a8823e5f1547f6d1096ae1e34fc5644a0fb375d5/RESTURANT_SENTIMENT/output.png)
## Model File

- The trained sentiment analysis model is saved as `sentiment_model.pkl`.
- This file can be loaded for making offline predictions without retraining.



## Future Enhancements

- Integration with real-time restaurant review platforms.
- Experimenting with deep learning models (LSTMs, Transformers) for improved accuracy.

## Contributing

Feel free to submit issues and pull requests if you want to contribute to this project.

## License

This project is licensed under the MIT License.

