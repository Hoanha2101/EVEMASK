from sklearn.metrics.pairwise import cosine_similarity

class SimilarityBlock:

    @staticmethod
    def base_cosine(final_class_ids, class_0_indices, recognizeDataVector_array, outputs):
        for i in range(outputs.shape[0]):
            query_vector = outputs[i].reshape(1, -1)
            cos_sims = cosine_similarity(query_vector, recognizeDataVector_array).flatten()
            for idx, sim in enumerate(cos_sims):
                if sim >= 0.7:
                    final_class_ids[class_0_indices[i].item()] = 1
                    break
        return final_class_ids

SimilarityMethod = SimilarityBlock.base_cosine
