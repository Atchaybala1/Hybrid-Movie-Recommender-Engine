def hybrid_recommendations(userId, title, num_recommendations=5):
    # 1. Fetch top candidates based on Content-Based Filtering (genre similarity)
    # We fetch more candidates than needed (e.g., 30) to filter them using CF
    content_candidates = get_recommendations(title, num_recommendations=30)

    if isinstance(content_candidates, str):
        return content_candidates

    # Get movieIds for our candidates
    candidates_df = movies_unique[movies_unique['title'].isin(content_candidates['title'])].copy()

    # 2. Extract Collaborative Filtering predictions for this specific user
    if userId not in preds_df.index:
        # Fallback to pure content-based if user doesn't exist
        return content_candidates.head(num_recommendations)

    user_predictions = preds_df.loc[userId]

    # Map predicted ratings to our candidates
    predicted_ratings = []
    for idx, row in candidates_df.iterrows():
        m_id = row['movieId']
        if m_id in user_predictions.index:
            predicted_ratings.append(user_predictions[m_id])
        else:
    # 3. Sort by predicted ratings
    hybrid_recs = candidates_df.sort_values(by='predicted_rating', ascending=False)

    return hybrid_recs[['title', 'genres', 'predicted_rating']].head(num_recommendations)

# Test the Hybrid Recommender for User 1 and Toy Story (1995)
test_user = 1
test_movie = "Toy Story (1995)"
print(f"Hybrid recommendations for User {test_user} based on their interest in '{test_movie}':")
display(hybrid_recommendations(test_user, test_movie, num_recommendations=5))
