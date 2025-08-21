"""
Performance Tests for EVEMASK Newsletter Backend
Author: EVEMASK Team
"""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import patch


class TestPerformanceMetrics:
    """Test cases for API performance metrics"""
    
    def test_newsletter_signup_response_time(self, client, temp_subscribers_file, mock_gmail_service):
        """Test newsletter signup response time"""
        valid_data = {"email": "performance@example.com"}
        
        start_time = time.time()
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.post("/newsletter/signup", json=valid_data)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_subscribers_retrieval_response_time(self, client, temp_subscribers_file, sample_subscribers):
        """Test subscribers retrieval response time"""
        # Create larger dataset for performance testing
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                "email": f"perf{i}@example.com",
                "timestamp": "2025-01-01T10:00:00",
                "status": "active"
            })
        
        with open(temp_subscribers_file, 'w') as f:
            json.dump(large_dataset, f)
        
        start_time = time.time()
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.get("/subscribers")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second for 100 records
        assert len(response.json()["subscribers"]) == 100
    
    def test_concurrent_requests_performance(self, client, temp_subscribers_file, mock_gmail_service):
        """Test performance under concurrent requests"""
        def make_signup_request(email_suffix):
            data = {"email": f"concurrent{email_suffix}@example.com"}
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                return client.post("/newsletter/signup", json=data)
        
        start_time = time.time()
        
        # Simulate 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_signup_request, i) for i in range(10)]
            responses = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should complete within reasonable time
        assert total_time < 5.0  # 10 concurrent requests within 5 seconds
        
        # Most requests should succeed (allowing for some duplicates)
        success_count = sum(1 for r in responses if r.status_code in [200, 409])
        assert success_count >= 8  # At least 8 out of 10 should succeed or be duplicates
    
    def test_memory_usage_stability(self, client, temp_subscribers_file, mock_gmail_service):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for i in range(50):
            data = {"email": f"memory{i}@example.com"}
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json=data)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB for 50 requests)
        assert memory_growth < 50 * 1024 * 1024  # 50 MB
    
    def test_file_io_performance(self, temp_subscribers_file):
        """Test file I/O performance with large datasets"""
        from main import save_subscriber_to_file, load_subscribers
        
        # Time saving multiple subscribers
        start_time = time.time()
        
        for i in range(100):
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                save_subscriber_to_file(f"fileio{i}@example.com")
        
        save_time = time.time() - start_time
        
        # Time loading subscribers
        start_time = time.time()
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        load_time = time.time() - start_time
        
        assert save_time < 5.0  # Should save 100 records within 5 seconds
        assert load_time < 1.0  # Should load 100 records within 1 second
        assert len(subscribers) == 100


class TestLoadTesting:
    """Test cases for load testing scenarios"""
    
    def test_rapid_sequential_requests(self, client, temp_subscribers_file, mock_gmail_service):
        """Test rapid sequential requests"""
        response_times = []
        
        for i in range(20):
            data = {"email": f"rapid{i}@example.com"}
            
            start_time = time.time()
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json=data)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            
            # Most requests should succeed
            assert response.status_code in [200, 409]  # Success or duplicate
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.5  # Average under 0.5 seconds
        
        # No single request should take too long
        max_response_time = max(response_times)
        assert max_response_time < 2.0  # Max 2 seconds
    
    def test_email_service_performance(self, mock_gmail_service):
        """Test email service performance"""
        from main import send_welcome_email
        
        email_send_times = []
        
        with patch('main.build', return_value=mock_gmail_service):
            for i in range(10):
                start_time = time.time()
                result = send_welcome_email(f"email{i}@example.com")
                end_time = time.time()
                
                email_send_times.append(end_time - start_time)
                assert result is True
        
        # Email sending should be fast (mocked)
        avg_email_time = sum(email_send_times) / len(email_send_times)
        assert avg_email_time < 0.1  # Very fast with mocked service
    
    def test_json_parsing_performance(self, temp_subscribers_file):
        """Test JSON parsing performance with large files"""
        # Create large JSON file
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                "email": f"json{i}@example.com",
                "timestamp": "2025-01-01T10:00:00",
                "status": "active"
            })
        
        with open(temp_subscribers_file, 'w') as f:
            json.dump(large_dataset, f)
        
        from main import load_subscribers
        
        # Time JSON parsing
        start_time = time.time()
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        parse_time = time.time() - start_time
        
        assert len(subscribers) == 1000
        assert parse_time < 0.5  # Should parse 1000 records within 0.5 seconds


class TestResourceUtilization:
    """Test cases for resource utilization"""
    
    def test_cpu_usage_under_load(self, client, temp_subscribers_file, mock_gmail_service):
        """Test CPU usage during high load"""
        import psutil
        
        # Monitor CPU usage
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # Generate load
        for i in range(30):
            data = {"email": f"cpu{i}@example.com"}
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json=data)
        
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        # CPU usage shouldn't spike excessively for simple operations
        cpu_increase = cpu_percent_after - cpu_percent_before
        assert cpu_increase < 50  # Less than 50% CPU increase
    
    def test_response_time_degradation(self, client, temp_subscribers_file, mock_gmail_service):
        """Test response time degradation with increasing load"""
        response_times = []
        
        # Test increasing dataset sizes
        for batch in range(5):
            # Add some existing data
            for i in range(batch * 20):
                data = {"email": f"batch{batch}_{i}@example.com"}
                with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                    client.post("/newsletter/signup", json=data)
            
            # Measure response time for new request
            test_data = {"email": f"test_batch{batch}@example.com"}
            
            start_time = time.time()
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json=test_data)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            assert response.status_code == 200
        
        # Response times shouldn't degrade significantly
        first_response = response_times[0]
        last_response = response_times[-1]
        
        # Last response should not be more than 3x slower than first
        assert last_response < first_response * 3
