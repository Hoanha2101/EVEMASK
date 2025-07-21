"""
Similarity computation utilities for the EVEMASK Pipeline system.
Provides cosine similarity-based matching and classification for feature vectors.
Handles similarity calculations between query vectors and reference data for object recognition.

Author: EVEMASK Team
Version: 1.0.0
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========================================================================
# SIMILARITY BLOCK CLASS
# ========================================================================
class SimilarityBlock:
    """
    Provides methods for computing cosine similarity between feature vectors.
    Used to classify unknown query vectors based on known reference vectors.
    """

    @staticmethod
    def base_cosine(final_class_ids: np.ndarray,
                    class_0_indices: np.ndarray,
                    recognizeDataVector_array: np.ndarray,
                    outputs: np.ndarray,
                    threshold: float = 0.65
                    ) -> np.ndarray:
        """
        Computes cosine similarity between query vectors and reference data to update class labels.

        Args:
            final_class_ids (np.ndarray): Array to store final class predictions.
            class_0_indices (np.ndarray): Indices of elements initially predicted as class 0 (unknown).
            recognizeDataVector_array (np.ndarray): Reference feature vectors for known classes.
            outputs (np.ndarray): Query feature vectors to classify.

        Returns:
            np.ndarray: Updated class IDs after similarity-based matching.

        Notes:
            - A similarity threshold of x is used to determine positive matches.
            - Any query vector with similarity >= x to any reference vector will be reclassified as class 1.
        """
        # Compute cosine similarity between query and reference vectors
        cos_sims = cosine_similarity(outputs, recognizeDataVector_array)  # (num_query, num_ref)
        # Identify query vectors with at least one match above threshold
        has_match = (cos_sims >= threshold).any(axis=1)  # shape: (num_query,)

        # Efficient vectorized update of class IDs for matched queries
        matched_indices = class_0_indices[has_match]
        final_class_ids[matched_indices] = 1

        return final_class_ids

# ========================================================================
# SIMILARITY METHOD ASSIGNMENT
# ========================================================================
# Assign the default similarity function for the pipeline
SimilarityMethod = SimilarityBlock.base_cosine
