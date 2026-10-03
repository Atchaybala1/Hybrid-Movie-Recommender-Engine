import pickle
import pandas as pd

class MovieRecommender:
    def __init__(self, model_path='hybrid_recommender.pkl'):
        # Load the pre-packaged model artifacts
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.movies_unique = self.model['movies_unique']
        self.indices = self.model['indices']
        self.cosine_sim = self.model['cosine_sim']
        self.preds_df = self.model['preds_df']

    def get_titles_list(self):
        """Helper to get a list of all unique movie titles for auto-complete."""
        return sorted(self.movies_unique['title'].tolist())

    def recommend(self, userId, title, num_recommendations=5):
        # 1. Content-Based candidates lookup
        if title not in self.indices:
            return f"Movie '{title}' not found."

        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if score[0] != idx][:30]

        movie_indices = [i[0] for i in sim_scores]
        content_candidates = self.movies_unique.iloc[movie_indices][['title', 'genres']]

        # 2. Collaborative filtering reranking
        candidates_df = self.movies_unique[self.movies_unique['title'].isin(content_candidates['title'])].copy()

        if userId not in self.preds_df.index:
            # Fallback if User ID doesn't exist
            return content_candidates.head(num_recommendations)

        user_predictions = self.preds_df.loc[userId]

        predicted_ratings = []
        for _, row in candidates_df.iterrows():
            m_id = row['movieId']
            predicted_ratings.append(user_predictions[m_id] if m_id in user_predictions.index else 0.0)

        candidates_df['predicted_rating'] = predicted_ratings
        hybrid_recs = candidates_df.sort_values(by='predicted_rating', ascending=False)

        return hybrid_recs[['title', 'genres', 'predicted_rating']].head(num_recommendations)
