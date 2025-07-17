"""
Similarity computation utilities for the EVEMASK Pipeline system.
Provides cosine similarity-based matching and classification for feature vectors.
Handles similarity calculations between query vectors and reference data for object recognition.

Author: EVEMASK Team
Version: 1.0.0
"""

from sklearn.metrics.pairwise import cosine_similarity

# ========================================================================
# SIMILARITY BLOCK CLASS
# ========================================================================
class SimilarityBlock:
    """
    The SimilarityBlock class provides methods for computing similarity between feature vectors.
    Uses cosine similarity to match query vectors against reference data for classification.
    """

    @staticmethod
    def base_cosine(final_class_ids, class_0_indices, recognizeDataVector_array, outputs):
        """
        Compute cosine similarity between query vectors and reference data to determine class assignments.
        
        Args:
            final_class_ids (torch.Tensor): Tensor to store final class assignments
            class_0_indices (torch.Tensor): Indices of objects classified as class 0 (unknown)
            recognizeDataVector_array (numpy.ndarray): Reference feature vectors for known classes
            outputs (torch.Tensor): Query feature vectors to be classified
            
        Returns:
            torch.Tensor: Updated final_class_ids with similarity-based classifications
            
        Note:
            Uses a threshold of x for cosine similarity to determine positive matches.
            If similarity >= x, the object is reclassified from class 0 to class 1.
        """
        # Process each query vector
        for i in range(outputs.shape[0]):
            # Reshape query vector for similarity computation
            query_vector = outputs[i].reshape(1, -1)
            # Compute cosine similarity between query and all reference vectors
            cos_sims = cosine_similarity(query_vector, recognizeDataVector_array).flatten()
            
            # Check if any reference vector has sufficient similarity
            for idx, sim in enumerate(cos_sims):
                if sim >= 0.7:  # Similarity threshold for positive match
                    # Reclassify from class 0 to class 1 (known class)
                    final_class_ids[class_0_indices[i].item()] = 1
                    break  # Stop after first match above threshold
                    
        return final_class_ids

# ========================================================================
# SIMILARITY METHOD ASSIGNMENT
# ========================================================================
# Assign the default similarity method for the pipeline
SimilarityMethod = SimilarityBlock.base_cosine
