# Movie-Recommendation-System
This project demonstrates a movie recommendation system using both traditional data science techniques and deep learning. It leverages the MovieLens dataset to provide personalized movie suggestions for users.
Key Features:
Exploratory Data Analysis (EDA)
Analyzes user ratings and movie metadata.
Visualizes rating distributions and top-rated movies.

Collaborative Filtering
Implements a User-Item Nearest Neighbors (KNN) model.
Recommends movies based on similar users’ preferences.

Deep Learning Autoencoder Recommender
Uses an autoencoder to learn latent features from the user-movie matrix.
Predicts personalized ratings and recommends unseen movies.

Visualization
Displays rating distributions with Seaborn and Matplotlib for intuitive understanding of the dataset.

Technologies & Libraries
Python 3.9
Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (KNN, MinMaxScaler, train/test split)
TensorFlow / Keras (Autoencoder)
