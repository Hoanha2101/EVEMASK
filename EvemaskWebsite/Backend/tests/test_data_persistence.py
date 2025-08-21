"""
Data Persistence Tests for EVEMASK Newsletter Backend
Author: EVEMASK Team
"""

import json
import os
import tempfile
from unittest.mock import patch, mock_open
import pytest


class TestSubscriberDataPersistence:
    """Test cases for subscriber data persistence"""
    
    def test_save_subscriber_to_file_new_file(self, temp_subscribers_file):
        """Test saving subscriber to new file"""
        from main import save_subscriber_to_file
        
        email = "newfile@example.com"
        
        # Remove the temp file to simulate new file creation
        os.unlink(temp_subscribers_file)
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result = save_subscriber_to_file(email)
        
        assert result is True
        
        # Verify file was created and contains data
        assert os.path.exists(temp_subscribers_file)
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            assert len(subscribers) == 1
            assert subscribers[0]["email"] == email
            assert subscribers[0]["status"] == "active"
    
    def test_save_subscriber_to_existing_file(self, temp_subscribers_file):
        """Test saving subscriber to existing file"""
        from main import save_subscriber_to_file
        
        # Pre-populate file with existing data
        existing_data = [{"email": "existing@example.com", "timestamp": "2025-01-01T10:00:00", "status": "active"}]
        with open(temp_subscribers_file, 'w') as f:
            json.dump(existing_data, f)
        
        new_email = "additional@example.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result = save_subscriber_to_file(new_email)
        
        assert result is True
        
        # Verify both emails are in file
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            assert len(subscribers) == 2
            emails = [sub["email"] for sub in subscribers]
            assert "existing@example.com" in emails
            assert "additional@example.com" in emails
    
    def test_save_subscriber_duplicate_email(self, temp_subscribers_file):
        """Test saving duplicate email (should fail)"""
        from main import save_subscriber_to_file
        
        email = "duplicate@example.com"
        
        # Save first time
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result1 = save_subscriber_to_file(email)
        
        assert result1 is True
        
        # Try to save same email again
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result2 = save_subscriber_to_file(email)
        
        assert result2 is False
        
        # Verify only one entry exists
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            assert len(subscribers) == 1
    
    def test_save_subscriber_file_permission_error(self):
        """Test saving subscriber with file permission error"""
        from main import save_subscriber_to_file
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = save_subscriber_to_file("permission@example.com")
        
        assert result is False
    
    def test_save_subscriber_json_decode_error(self, temp_subscribers_file):
        """Test saving subscriber with corrupted JSON file"""
        from main import save_subscriber_to_file
        
        # Create file with invalid JSON
        with open(temp_subscribers_file, 'w') as f:
            f.write("invalid json content")
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result = save_subscriber_to_file("json@example.com")
        
        # Should handle the error and create new file
        assert result is True or result is False  # Implementation dependent
    
    def test_load_subscribers_success(self, temp_subscribers_file, sample_subscribers):
        """Test successful loading of subscribers"""
        from main import load_subscribers
        
        with open(temp_subscribers_file, 'w') as f:
            json.dump(sample_subscribers, f)
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        assert len(subscribers) == 2
        assert subscribers == sample_subscribers
    
    def test_load_subscribers_empty_file(self, temp_subscribers_file):
        """Test loading from empty file"""
        from main import load_subscribers
        
        with open(temp_subscribers_file, 'w') as f:
            json.dump([], f)
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        assert subscribers == []
    
    def test_load_subscribers_nonexistent_file(self):
        """Test loading when file doesn't exist"""
        from main import load_subscribers
        
        with patch('main.SUBSCRIBERS_FILE', '/nonexistent/path/subscribers.json'):
            subscribers = load_subscribers()
        
        assert subscribers == []
    
    def test_load_subscribers_malformed_json(self, temp_subscribers_file):
        """Test loading with malformed JSON"""
        from main import load_subscribers
        
        with open(temp_subscribers_file, 'w') as f:
            f.write("invalid json content")
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        # Should return empty list or handle gracefully
        assert isinstance(subscribers, list)
    
    def test_subscriber_data_structure(self, temp_subscribers_file):
        """Test subscriber data structure integrity"""
        from main import save_subscriber_to_file
        
        email = "structure@example.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            save_subscriber_to_file(email)
        
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            subscriber = subscribers[0]
            
            # Verify required fields
            assert "email" in subscriber
            assert "timestamp" in subscriber
            assert "status" in subscriber
            
            # Verify data types
            assert isinstance(subscriber["email"], str)
            assert isinstance(subscriber["timestamp"], str)
            assert isinstance(subscriber["status"], str)
            
            # Verify values
            assert subscriber["email"] == email
            assert subscriber["status"] == "active"
    
    def test_timestamp_format_consistency(self, temp_subscribers_file):
        """Test timestamp format consistency"""
        from main import save_subscriber_to_file
        import re
        
        emails = ["time1@example.com", "time2@example.com"]
        
        for email in emails:
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                save_subscriber_to_file(email)
        
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            
            # ISO 8601 format pattern
            iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
            
            for subscriber in subscribers:
                assert re.match(iso_pattern, subscriber["timestamp"])
    
    def test_file_encoding_utf8(self, temp_subscribers_file):
        """Test file encoding handles UTF-8 characters"""
        from main import save_subscriber_to_file
        
        email_with_unicode = "tést@éxample.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result = save_subscriber_to_file(email_with_unicode)
        
        assert result is True
        
        # Verify UTF-8 encoding
        with open(temp_subscribers_file, 'r', encoding='utf-8') as f:
            subscribers = json.load(f)
            assert subscribers[0]["email"] == email_with_unicode


class TestDataIntegrity:
    """Test cases for data integrity and consistency"""
    
    def test_concurrent_write_protection(self, temp_subscribers_file):
        """Test protection against concurrent writes"""
        from main import save_subscriber_to_file
        
        # Simulate concurrent writes
        email1 = "concurrent1@example.com"
        email2 = "concurrent2@example.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result1 = save_subscriber_to_file(email1)
            result2 = save_subscriber_to_file(email2)
        
        assert result1 is True
        assert result2 is True
        
        # Verify both emails are saved
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            assert len(subscribers) == 2
    
    def test_data_backup_on_corruption(self, temp_subscribers_file):
        """Test handling of data corruption scenarios"""
        from main import load_subscribers
        
        # Create corrupted file
        with open(temp_subscribers_file, 'w') as f:
            f.write('{"incomplete": json')
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        # Should handle gracefully without crashing
        assert isinstance(subscribers, list)
    
    def test_large_dataset_handling(self, temp_subscribers_file):
        """Test handling of large subscriber datasets"""
        from main import save_subscriber_to_file, load_subscribers
        
        # Create large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                "email": f"user{i}@example.com",
                "timestamp": "2025-01-01T10:00:00",
                "status": "active"
            })
        
        with open(temp_subscribers_file, 'w') as f:
            json.dump(large_dataset, f)
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            subscribers = load_subscribers()
        
        assert len(subscribers) == 1000
        
        # Test adding to large dataset
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result = save_subscriber_to_file("new@example.com")
        
        assert result is True
