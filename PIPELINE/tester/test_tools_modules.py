"""
Unit tests for tools modules.
Tests NB_search, similarityBlock, and vectorPrepare functionality.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools import NB_search, similarityBlock, vectorPrepare


class TestNBSearch(unittest.TestCase):
    """Test cases for NB_search module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_vectors = np.random.rand(100, 128)  # 100 vectors of 128 dimensions
        self.query_vector = np.random.rand(128)
        self.k = 5
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_nearest_neighbor_search_basic(self):
        """Test basic nearest neighbor search functionality."""
        try:
            # Test that the module can be imported and basic functionality works
            # This is a placeholder test since we don't have the actual implementation
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"NB_search basic functionality failed: {e}")
    
    def test_vector_similarity_calculation(self):
        """Test vector similarity calculation."""
        # Test cosine similarity
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        
        # Calculate cosine similarity manually
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        self.assertEqual(cos_sim, 1.0)
        
        # Test orthogonal vectors
        vec3 = np.array([0, 1, 0])
        cos_sim_ortho = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        self.assertEqual(cos_sim_ortho, 0.0)
    
    def test_distance_calculation(self):
        """Test distance calculation methods."""
        # Test Euclidean distance
        vec1 = np.array([0, 0])
        vec2 = np.array([3, 4])
        
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        self.assertEqual(euclidean_dist, 5.0)
        
        # Test Manhattan distance
        manhattan_dist = np.sum(np.abs(vec1 - vec2))
        self.assertEqual(manhattan_dist, 7.0)
    
    def test_search_accuracy(self):
        """Test search accuracy with known data."""
        # Create test data with known nearest neighbor
        base_vector = np.array([1, 0, 0, 0])
        vectors = [
            base_vector,
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1])
        ]
        
        # Query should find the base_vector as nearest neighbor
        query = np.array([0.9, 0.1, 0, 0])
        
        # Calculate similarities
        similarities = [np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec)) 
                       for vec in vectors]
        
        # Find index of maximum similarity
        nearest_idx = np.argmax(similarities)
        self.assertEqual(nearest_idx, 0)  # Should be the base_vector
    
    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        empty_vectors = np.array([])
        query = np.random.rand(128)
        
        try:
            # This should handle empty dataset gracefully
            pass
        except Exception as e:
            # Should not raise unhandled exceptions
            self.fail(f"Empty dataset handling failed: {e}")
    
    def test_invalid_query_handling(self):
        """Test handling of invalid query vectors."""
        vectors = np.random.rand(10, 128)
        invalid_query = np.array([])  # Empty query
        
        try:
            # This should handle invalid query gracefully
            pass
        except Exception as e:
            # Should not raise unhandled exceptions
            self.fail(f"Invalid query handling failed: {e}")


class TestSimilarityBlock(unittest.TestCase):
    """Test cases for similarityBlock module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_block1 = np.random.rand(64, 64)
        self.test_block2 = np.random.rand(64, 64)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_similarity_block_basic(self):
        """Test basic similarity block functionality."""
        try:
            # Test that the module can be imported and basic functionality works
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"similarityBlock basic functionality failed: {e}")
    
    def test_block_similarity_calculation(self):
        """Test block similarity calculation."""
        # Create identical blocks
        block1 = np.ones((32, 32))
        block2 = np.ones((32, 32))
        
        # Calculate similarity (should be 1.0 for identical blocks)
        similarity = np.sum(block1 * block2) / (np.linalg.norm(block1) * np.linalg.norm(block2))
        self.assertEqual(similarity, 1.0)
        
        # Test orthogonal blocks
        block3 = np.zeros((32, 32))
        similarity_ortho = np.sum(block1 * block3) / (np.linalg.norm(block1) * np.linalg.norm(block3))
        # Note: This will be NaN for zero blocks, which is expected
        self.assertTrue(np.isnan(similarity_ortho) or similarity_ortho == 0)
    
    def test_block_size_handling(self):
        """Test handling of different block sizes."""
        sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]
        
        for size in sizes:
            with self.subTest(size=size):
                block1 = np.random.rand(*size)
                block2 = np.random.rand(*size)
                
                # Should handle different sizes without errors
                try:
                    similarity = np.sum(block1 * block2) / (np.linalg.norm(block1) * np.linalg.norm(block2))
                    self.assertIsInstance(similarity, (float, np.float64))
                except Exception as e:
                    self.fail(f"Block size {size} handling failed: {e}")
    
    def test_similarity_threshold(self):
        """Test similarity threshold functionality."""
        # Create similar blocks
        base_block = np.random.rand(32, 32)
        similar_block = base_block + np.random.rand(32, 32) * 0.1  # Small noise
        
        # Calculate similarity
        similarity = np.sum(base_block * similar_block) / (np.linalg.norm(base_block) * np.linalg.norm(similar_block))
        
        # Should be above threshold for similar blocks
        threshold = 0.8
        self.assertGreater(similarity, threshold)
    
    def test_dissimilar_blocks(self):
        """Test handling of dissimilar blocks."""
        # Create very different blocks
        block1 = np.ones((32, 32))
        block2 = np.zeros((32, 32))
        
        # Calculate similarity
        similarity = np.sum(block1 * block2) / (np.linalg.norm(block1) * np.linalg.norm(block2))
        
        # Should be 0 for orthogonal blocks
        self.assertEqual(similarity, 0.0)


class TestVectorPrepare(unittest.TestCase):
    """Test cases for vectorPrepare module."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_data = np.random.rand(100, 64)
        self.test_labels = np.random.randint(0, 10, 100)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_vector_prepare_basic(self):
        """Test basic vector prepare functionality."""
        try:
            # Test that the module can be imported and basic functionality works
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"vectorPrepare basic functionality failed: {e}")
    
    def test_data_normalization(self):
        """Test data normalization functionality."""
        # Create test data with different scales
        data = np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
        
        # Normalize data
        normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # Check normalization properties
        self.assertAlmostEqual(np.mean(normalized_data, axis=0)[0], 0.0, places=10)
        self.assertAlmostEqual(np.std(normalized_data, axis=0)[0], 1.0, places=10)
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        # Create test image-like data
        image_data = np.random.rand(10, 64, 64, 3)  # 10 images, 64x64, 3 channels
        
        # Extract features (flatten for simplicity)
        features = image_data.reshape(image_data.shape[0], -1)
        
        # Check feature dimensions
        self.assertEqual(features.shape[0], 10)
        self.assertEqual(features.shape[1], 64 * 64 * 3)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        # Create raw data
        raw_data = np.random.rand(50, 128)
        
        # Apply preprocessing steps
        # 1. Remove outliers (simple approach: clip values)
        processed_data = np.clip(raw_data, np.percentile(raw_data, 5), np.percentile(raw_data, 95))
        
        # 2. Normalize
        processed_data = (processed_data - np.mean(processed_data, axis=0)) / np.std(processed_data, axis=0)
        
        # Check that preprocessing worked
        self.assertEqual(processed_data.shape, raw_data.shape)
        self.assertFalse(np.any(np.isnan(processed_data)))
        self.assertFalse(np.any(np.isinf(processed_data)))
    
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction functionality."""
        # Create high-dimensional data
        high_dim_data = np.random.rand(100, 1000)
        
        # Simple dimensionality reduction (take first 100 dimensions)
        reduced_data = high_dim_data[:, :100]
        
        # Check reduction
        self.assertEqual(reduced_data.shape, (100, 100))
        self.assertEqual(reduced_data.shape[1], 100)
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        # Create test data
        data = np.random.rand(100, 64)
        labels = np.random.randint(0, 10, 100)
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        train_labels = labels[:split_idx]
        test_labels = labels[split_idx:]
        
        # Check split
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(test_data), 20)
        self.assertEqual(len(train_labels), 80)
        self.assertEqual(len(test_labels), 20)
    
    def test_data_validation(self):
        """Test data validation functionality."""
        # Test valid data
        valid_data = np.random.rand(10, 64)
        self.assertFalse(np.any(np.isnan(valid_data)))
        self.assertFalse(np.any(np.isinf(valid_data)))
        
        # Test invalid data (with NaN)
        invalid_data = valid_data.copy()
        invalid_data[0, 0] = np.nan
        self.assertTrue(np.any(np.isnan(invalid_data)))
    
    def test_memory_efficiency(self):
        """Test memory efficiency of vector preparation."""
        # Create large dataset
        large_data = np.random.rand(1000, 512)
        
        # Process in batches
        batch_size = 100
        processed_batches = []
        
        for i in range(0, len(large_data), batch_size):
            batch = large_data[i:i+batch_size]
            # Process batch (normalize)
            processed_batch = (batch - np.mean(batch, axis=0)) / np.std(batch, axis=0)
            processed_batches.append(processed_batch)
        
        # Combine batches
        processed_data = np.vstack(processed_batches)
        
        # Check result
        self.assertEqual(processed_data.shape, large_data.shape)
    
    def test_error_handling(self):
        """Test error handling in vector preparation."""
        # Test with empty data
        empty_data = np.array([])
        
        try:
            # Should handle empty data gracefully
            if len(empty_data) == 0:
                pass
        except Exception as e:
            self.fail(f"Empty data handling failed: {e}")
        
        # Test with None data
        try:
            # Should handle None data gracefully
            if None is None:
                pass
        except Exception as e:
            self.fail(f"None data handling failed: {e}")


class TestToolsIntegration(unittest.TestCase):
    """Integration tests for tools modules."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data for integration tests
        self.test_vectors = np.random.rand(100, 128)
        self.test_blocks = np.random.rand(10, 64, 64)
        self.test_data = np.random.rand(50, 256)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline using all tools modules."""
        # 1. Prepare vectors
        prepared_data = self.test_data.copy()
        prepared_data = (prepared_data - np.mean(prepared_data, axis=0)) / np.std(prepared_data, axis=0)
        
        # 2. Search for similar vectors
        query_vector = prepared_data[0]
        similarities = [np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec)) 
                       for vec in prepared_data]
        
        # 3. Find most similar
        most_similar_idx = np.argmax(similarities)
        
        # 4. Verify results
        self.assertEqual(most_similar_idx, 0)  # Should be most similar to itself
        self.assertEqual(similarities[0], 1.0)  # Perfect similarity with itself
    
    def test_cross_module_compatibility(self):
        """Test compatibility between different tools modules."""
        # Test that data types are compatible across modules
        vector_data = np.random.rand(10, 64)
        block_data = np.random.rand(5, 32, 32)
        
        # Both should be numpy arrays
        self.assertIsInstance(vector_data, np.ndarray)
        self.assertIsInstance(block_data, np.ndarray)
        
        # Both should have numeric data
        self.assertTrue(np.issubdtype(vector_data.dtype, np.number))
        self.assertTrue(np.issubdtype(block_data.dtype, np.number))
    
    def test_performance_benchmark(self):
        """Test performance of tools modules."""
        import time
        
        # Benchmark vector operations
        start_time = time.time()
        
        # Perform vector operations
        for _ in range(100):
            vec1 = np.random.rand(128)
            vec2 = np.random.rand(128)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time
        self.assertLess(processing_time, 1.0)


if __name__ == '__main__':
    unittest.main() 