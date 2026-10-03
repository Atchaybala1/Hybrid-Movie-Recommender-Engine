import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Create a User-Item Matrix
user_ratings_pivot = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Convert to numpy array
R = user_ratings_pivot.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Convert to sparse matrix
_R_demeaned_sparse = csr_matrix(R_demeaned)

# Perform Singular Value Decomposition (SVD)
# We'll extract 20 latent features
U, sigma, Vt = svds(_R_demeaned_sparse, k=20)

# Convert sigma to diagonal matrix
sigma = np.diag(sigma)

# Reconstruct the predicted ratings matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# Convert to DataFrame
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_ratings_pivot.columns, index=user_ratings_pivot.index)

print("Predicted ratings matrix shape:", preds_df.shape)
