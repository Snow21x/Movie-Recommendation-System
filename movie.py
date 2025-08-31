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

#Loads Dataset
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

#Data Cleaning & EDA
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


#Collaborative Filtering

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.values)
def recommend_knn(user_id, n_recommendations=5):
    user_idx = user_id - 1
    distances, indices = model_knn.kneighbors(
        user_movie_matrix.values[user_idx].reshape(1, -1),
        n_neighbors=n_recommendations+1
    )
    rec_indices = indices.flatten()[1:]
    rec_movies = user_movie_matrix.index[rec_indices]
    return rec_movies
print("Collaborative Filtering Recommendations for User 1:")
print(recommend_knn(1))


#Deep Learning Autoencoder Recommender

# Build Autoencoder
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
    epochs=20,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)
user_idx = 0
predicted_ratings = autoencoder.predict(ratings_matrix_scaled[user_idx].reshape(1, -1)).flatten()
already_rated = user_movie_matrix.values[user_idx] > 0
predicted_ratings[already_rated] = 0
top_indices = predicted_ratings.argsort()[::-1][:5]
recommended_movies = movies.iloc[top_indices]['title'].values
print("\nDeep Learning Recommendations for User 1:")
print(recommended_movies)

# Genre Preferences
movies['genres'] = movies['genres'].str.split('|')
def get_genres_for_recommendations(movie_ids):
    rec_genres = movies[movies['movieId'].isin(movie_ids)]['genres']
    all_genres = [g for sublist in rec_genres for g in sublist]
    return pd.Series(all_genres).value_counts()

print("\nTop Genres for User 1 (based on Autoencoder Recommendations):")
print(get_genres_for_recommendations(recommended_ids).head(5))

