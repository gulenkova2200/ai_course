import pickle
import numpy as np
import pandas as pd

class UserCFRecommender:
    def __init__(self, model_path: str = None):
        if model_path:
            self.load(model_path)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.user_similarity_matrix = model_data['user_similarity_matrix']
        self.user_item_matrix = model_data['user_ratings_matrix']
        self.all_movies = model_data['all_movies']
        
        self.movie_ids = self.user_item_matrix.columns.tolist()
        self.user_ids = self.user_item_matrix.index.tolist()
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        print(f"Модель загружена. Пользователей: {len(self.user_ids)}, Фильмов: {len(self.movie_ids)}")
        print(f"Shape similarity matrix: {self.user_similarity_matrix.shape}")
        print(f"Shape ratings matrix: {self.user_item_matrix.shape}")
    
    def predict(self, user_id, movie_id):
        try:
            if user_id not in self.user_item_matrix.index:
                print(f"User {user_id} not found, returning default rating")
                return 3.0
            
            user_idx = self.user_to_idx[user_id]

            if self.user_similarity_matrix.ndim == 2:
                similarity_row = self.user_similarity_matrix[user_idx]
            else:
                print(f"Unexpected similarity matrix shape: {self.user_similarity_matrix.shape}")
                return 3.0
            
            if movie_id not in self.user_item_matrix.columns:
                print(f"Movie {movie_id} not found, returning user average")
                user_ratings = self.user_item_matrix.loc[user_id]
                return user_ratings[user_ratings > 0].mean() if (user_ratings > 0).any() else 3.0
            
            movie_ratings = self.user_item_matrix[movie_id].values
            mask = movie_ratings > 0
            if not mask.any():
                user_ratings = self.user_item_matrix.loc[user_id]
                return user_ratings[user_ratings > 0].mean() if (user_ratings > 0).any() else 3.0
            
            similarities = similarity_row[mask]
            ratings = movie_ratings[mask]
            
            if similarities.sum() == 0:
                return ratings.mean()
            
            pred = np.average(ratings, weights=similarities)
            return np.clip(pred, 0.5, 5.0)
            
        except Exception as e:
            print(f"Error in predict: {e}")
            return 3.0
    
    def recommend(self, user_id, k=10):
        try:
            user_id = int(user_id)
            
            if user_id not in self.user_item_matrix.index:
                print(f"User {user_id} not found, returning popular movies")
                return self._get_popular_movies(k)
            
            user_idx = self.user_to_idx[user_id]
            
            if isinstance(self.user_similarity_matrix, pd.DataFrame):
                similarity_row = self.user_similarity_matrix.iloc[user_idx].values
            else:
                similarity_row = self.user_similarity_matrix[user_idx]
            
            ratings_matrix = self.user_item_matrix.values
            
            weighted_ratings = ratings_matrix.T * similarity_row
            total_scores = weighted_ratings.sum(axis=1)
            
            sum_weights = similarity_row.sum()
            if sum_weights > 0:
                total_scores = total_scores / sum_weights
            
            user_ratings = self.user_item_matrix.loc[user_id].values
            
            already_rated_mask = user_ratings > 0
            total_scores[already_rated_mask] = -1
            
            top_indices = np.argsort(total_scores)[::-1][:k]
            recommendations = [self.movie_ids[idx] for idx in top_indices if total_scores[idx] > 0]
            
            if not recommendations:
                print(f"No recommendations for user {user_id}, returning popular movies")
                return self._get_popular_movies(k)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in recommend: {e}")
            import traceback
            traceback.print_exc()
            return self._get_popular_movies(k)
    
    def _get_popular_movies(self, k=10):
        try:
            movie_popularity = self.user_item_matrix.astype(bool).sum(axis=0)
            popular = movie_popularity.sort_values(ascending=False)
            return popular.head(k).index.tolist()
        except Exception as e:
            print(f"Error getting popular movies: {e}")
            return self.movie_ids[:k] if self.movie_ids else []

    def get_user_info(self, user_id):
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_movies = user_ratings[user_ratings > 0]
            return {
                'exists': True,
                'rated_count': len(rated_movies),
                'avg_rating': rated_movies.mean() if len(rated_movies) > 0 else 0,
                'rated_movies': rated_movies.head(10).to_dict()
            }
        return {'exists': False}