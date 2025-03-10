from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open("model/sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/cv_model.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review_vector = vectorizer.transform([review]).toarray()  # Now using the correct vectorizer
        prediction = model.predict(review_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
