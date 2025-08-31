import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# Load Dataset
movies = pd.read_csv(
    'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat',
    sep='::', header=None, engine='python',
    names=['movieId','title','genres']
)
ratings = pd.read_csv(
    'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat',
    sep='::', header=None, engine='python',
    names=['userId','movieId','rating','timestamp']
)


# Data Cleaning & EDA
print("Movies head:\n", movies.head())
print("Ratings head:\n", ratings.head())
print("Number of unique users:", ratings['userId'].nunique())
print("Number of unique movies:", ratings['movieId'].nunique())
print("Ratings distribution:\n", ratings['rating'].value_counts())


# Visualization
plt.figure(figsize=(8,6))
sns.countplot(x='rating', data=ratings)
plt.title('Rating Distribution')
plt.show()
top_movies = ratings.groupby('movieId').size().sort_values(ascending=False).head(10)
top_movie_titles = movies[movies['movieId'].isin(top_movies.index)]
print("Top 10 rated movies:\n", top_movie_titles[['title']])


# Collaborative Filtering (with subset to prevent overflow)
filtered_ratings = ratings[
    (ratings['userId'] <= 5000) & (ratings['movieId'] <= 5000)
]

user_movie_matrix = filtered_ratings.pivot(
    index='userId', columns='movieId', values='rating'
).fillna(0)
print("Shape of user-movie matrix:", user_movie_matrix.shape)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.values)
def recommend_knn(user_id, n_recommendations=5):
    if user_id not in user_movie_matrix.index:
        print(f"User {user_id} not in dataset (try <= 5000).")
        return []
    user_idx = list(user_movie_matrix.index).index(user_id)
    distances, indices = model_knn.kneighbors(
        user_movie_matrix.values[user_idx].reshape(1, -1),
        n_neighbors=n_recommendations+1
    )
    rec_indices = indices.flatten()[1:] 
    rec_users = user_movie_matrix.index[rec_indices]
    return rec_users
print("Collaborative Filtering Recommendations for User 1:")
print(recommend_knn(1))


# Deep Learning Autoencoder Recommender
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Normalize ratings
scaler = MinMaxScaler()
ratings_matrix_scaled = scaler.fit_transform(user_movie_matrix.values)
X_train, X_test = train_test_split(ratings_matrix_scaled, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
autoencoder = build_autoencoder(input_dim)
autoencoder.fit(
    X_train, X_train,
    epochs=10,  
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# Predict recommendations for a given user
user_idx = 0  
predicted_ratings = autoencoder.predict(ratings_matrix_scaled[user_idx].reshape(1, -1)).flatten()
already_rated = user_movie_matrix.values[user_idx] > 0
predicted_ratings[already_rated] = 0
top_indices = predicted_ratings.argsort()[::-1][:5]
recommended_movies = movies[movies['movieId'].isin(user_movie_matrix.columns[top_indices])]['title'].values
print("\nDeep Learning Recommendations for User 1:")
print(recommended_movies)


