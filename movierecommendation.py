from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load the built-in MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Choose an algorithm (KNN basic) and train the model
model_knn = KNNBasic(sim_options={'user_based': True})
model_knn.fit(trainset)

# Make predictions on the test set
predictions_knn = model_knn.test(testset)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model
rmse_knn = rmse(predictions_knn)
print('RMSE for KNN:', rmse_knn)

# Alternatively, train an SVD algorithm
model_svd = SVD()
model_svd.fit(trainset)
predictions_svd = model_svd.test(testset)
rmse_svd = rmse(predictions_svd)
print('RMSE for SVD:', rmse_svd)

# Make recommendations for a user
user_id = str(196)  # Example user ID
user_movies = set(data.raw_ratings[i][0] for i in range(len(data.raw_ratings)) if data.raw_ratings[i][0] == user_id)
unseen_movies = [movie_id for movie_id in data.raw_ratings if movie_id[0] != user_id and movie_id[1] not in user_movies]

# Predict ratings for unseen movies
predictions = []
for movie_id in unseen_movies:
    movie_prediction = model_knn.predict(user_id, movie_id[1])
    predictions.append((movie_id[1], movie_prediction.est))

# Sort the predictions
top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

print("Top 10 Recommendations for User", user_id)
for movie_id, rating in top_recommendations:
    print("Movie ID:", movie_id, "| Rating:", round(rating, 2))
