import pickle as pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load movie data and similarity matrix
movies = pickle.load(open("models/movies.pkl", "rb"))
similarity = pickle.load(open("models/similarity.pkl", "rb"))

# Convert movies to DataFrame
df1 = pd.DataFrame(movies)

# Function to recommend movies
def recommend(movie_name):
    if movie_name not in df1["original_title"].values:
        return []
    index = df1[df1["original_title"] == movie_name].index[0]
    distances = similarity[index]
    movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [df1.iloc[i[0]].original_title for i in movie_indices]

@app.route('/')
def home():
    return render_template('index.html', movies=df1["original_title"].tolist())

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.json
    movie_name = data.get("movie")
    recommendations = recommend(movie_name)
    return jsonify({"recommended_movies": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
