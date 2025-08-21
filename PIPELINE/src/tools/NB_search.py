"""
Neighborhood-based search and blending utilities for the EVEMASK Pipeline system.
Provides FAISS-based similarity search and iterative neighborhood blending for feature vector enhancement.
Handles nearest neighbor search and weighted blending of similar feature vectors.

Author: EVEMASK Team
Version: 1.0.0
"""

import faiss
from sklearn.preprocessing import normalize
import numpy as np

# ========================================================================
# NEIGHBORHOOD SEARCH FUNCTION
# ========================================================================
def neighborhood_search(emb, thresh, k_neighbors):
    """
    Perform neighborhood search using FAISS to find similar feature vectors.
    
    Args:
        emb (numpy.ndarray): Input feature vectors [N, D]
        thresh (float): Similarity threshold for filtering neighbors
        k_neighbors (int): Maximum number of neighbors to search for
        
    Returns:
        tuple: (pred_index, pred_sim)
            - pred_index: List of neighbor indices for each vector
            - pred_sim: List of similarity scores for each vector
    """
    # Create FAISS index for inner product search
    index = faiss.IndexFlatIP(emb.shape[1])
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(emb)
    
    # Add vectors to the index
    index.add(emb)
    
    # Search for k nearest neighbors
    sim, I = index.search(emb, k_neighbors)
    
    # Filter results based on similarity threshold
    pred_index = []
    pred_sim = []
    
    for i in range(emb.shape[0]):
        cut_index = 0
        # Count neighbors above threshold
        for j in sim[i]:
            if j > thresh:
                cut_index += 1
            else:
                break
              
        # Store filtered indices and similarities
        pred_index.append(I[i][:(cut_index)])
        pred_sim.append(sim[i][:(cut_index)])
       
    return pred_index, pred_sim

# ========================================================================
# NEIGHBORHOOD BLENDING FUNCTION
# ========================================================================
def blend_neighborhood(emb, match_index_lst, similarities_lst):
    """
    Blend feature vectors with their similar neighbors using weighted averaging.
    
    Args:
        emb (numpy.ndarray): Original feature vectors [N, D]
        match_index_lst (list): List of neighbor indices for each vector
        similarities_lst (list): List of similarity scores for each vector
        
    Returns:
        numpy.ndarray: Blended feature vectors [N, D]
    """
    new_emb = emb.copy()
    
    for i in range(emb.shape[0]):
        # Get similar vectors for current vector
        cur_emb = emb[match_index_lst[i]]
        
        # Use similarities as weights for blending
        weights = np.expand_dims(similarities_lst[i], 1)
        
        # Weighted average of similar vectors
        new_emb[i] = (cur_emb * weights).sum(axis=0)
        
    # Normalize the blended vectors
    new_emb = normalize(new_emb, axis=1)
    
    return new_emb

# ========================================================================
# ITERATIVE NEIGHBORHOOD BLENDING
# ========================================================================
def iterative_neighborhood_blending(emb, threshes, k_neighbors):
    """
    Perform iterative neighborhood blending with multiple similarity thresholds.
    
    Args:
        emb (numpy.ndarray): Input feature vectors [N, D]
        threshes (list): List of similarity thresholds for each iteration
        k_neighbors (int): Maximum number of neighbors to search for
        
    Returns:
        tuple: (emb, match_index_lst, similarities_lst)
            - emb: Final blended feature vectors
            - match_index_lst: Final neighbor indices
            - similarities_lst: Final similarity scores
    """
    for thresh in threshes:
        # Find neighbors for current threshold
        match_index_lst, similarities_lst = neighborhood_search(emb, thresh, k_neighbors)
        
        # Blend vectors with their neighbors
        emb = blend_neighborhood(emb, match_index_lst, similarities_lst)
        
    return emb, match_index_lst, similarities_lst