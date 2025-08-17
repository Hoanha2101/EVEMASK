"""
API Endpoint Tests for EVEMASK Newsletter Backend
Author: EVEMASK Team
"""

import pytest
import json
from unittest.mock import patch, mock_open
from fastapi import status


class TestNewsletterSignupAPI:
    """Test cases for newsletter signup endpoint"""
    
    def test_newsletter_signup_success(self, client, mock_gmail_service, temp_subscribers_file):
        """Test successful newsletter signup"""
        valid_data = {"email": "newuser@example.com"}
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.post("/newsletter/signup", json=valid_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Successfully subscribed to newsletter!"
        assert response_data["email"] == "newuser@example.com"
        
        # Verify email was saved to file
        with open(temp_subscribers_file, 'r') as f:
            subscribers = json.load(f)
            assert len(subscribers) == 1
            assert subscribers[0]["email"] == "newuser@example.com"
            assert subscribers[0]["status"] == "active"
    
    def test_newsletter_signup_duplicate_email(self, client, temp_subscribers_file):
        """Test signup with duplicate email"""
        existing_data = [{"email": "existing@example.com", "timestamp": "2025-01-01T10:00:00", "status": "active"}]
        
        with open(temp_subscribers_file, 'w') as f:
            json.dump(existing_data, f)
        
        duplicate_data = {"email": "existing@example.com"}
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.post("/newsletter/signup", json=duplicate_data)
        
        assert response.status_code == status.HTTP_409_CONFLICT
        
        response_data = response.json()
        assert "already subscribed" in response_data["detail"].lower()
    
    @pytest.mark.parametrize("invalid_email", [
        "invalid-email",
        "",
        "test@",
        "@example.com",
        "test.example.com",
        "test@.com",
        "test@com",
        "test@@example.com"
    ])
    def test_newsletter_signup_invalid_email(self, client, invalid_email):
        """Test signup with invalid email formats"""
        invalid_data = {"email": invalid_email}
        
        response = client.post("/newsletter/signup", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_newsletter_signup_missing_email(self, client):
        """Test signup with missing email field"""
        response = client.post("/newsletter/signup", json={})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_newsletter_signup_empty_payload(self, client):
        """Test signup with empty payload"""
        response = client.post("/newsletter/signup")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_newsletter_signup_file_permission_error(self, client, mock_gmail_service):
        """Test signup when file cannot be written"""
        valid_data = {"email": "test@example.com"}
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            response = client.post("/newsletter/signup", json=valid_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        response_data = response.json()
        assert "error" in response_data["detail"].lower()
    
    def test_newsletter_signup_gmail_service_error(self, client, temp_subscribers_file):
        """Test signup when Gmail service fails"""
        valid_data = {"email": "test@example.com"}
        
        with patch('main.send_welcome_email', side_effect=Exception("Gmail API error")):
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json=valid_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestSubscribersRetrievalAPI:
    """Test cases for subscribers retrieval endpoint"""
    
    def test_get_subscribers_success(self, client, temp_subscribers_file, sample_subscribers):
        """Test successful retrieval of subscribers"""
        with open(temp_subscribers_file, 'w') as f:
            json.dump(sample_subscribers, f)
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.get("/subscribers")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert len(response_data["subscribers"]) == 2
        assert response_data["total"] == 2
        
        # Verify subscriber data
        emails = [sub["email"] for sub in response_data["subscribers"]]
        assert "user1@example.com" in emails
        assert "user2@example.com" in emails
    
    def test_get_subscribers_empty_file(self, client, temp_subscribers_file):
        """Test retrieval when no subscribers exist"""
        with open(temp_subscribers_file, 'w') as f:
            json.dump([], f)
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.get("/subscribers")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["subscribers"] == []
        assert response_data["total"] == 0
    
    def test_get_subscribers_file_not_found(self, client):
        """Test retrieval when subscribers file doesn't exist"""
        with patch('main.SUBSCRIBERS_FILE', '/nonexistent/path/subscribers.json'):
            response = client.get("/subscribers")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["subscribers"] == []
        assert response_data["total"] == 0
    
    def test_get_subscribers_malformed_json(self, client, temp_subscribers_file):
        """Test retrieval with malformed JSON file"""
        with open(temp_subscribers_file, 'w') as f:
            f.write("invalid json content")
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.get("/subscribers")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestHealthCheckAPI:
    """Test cases for health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test health check endpoint"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["service"] == "EVEMASK Newsletter API"
        assert "timestamp" in response_data


class TestCORSHeaders:
    """Test CORS headers and options"""
    
    def test_cors_preflight_request(self, client):
        """Test CORS preflight request"""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = client.options("/newsletter/signup", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]
    
    def test_cors_actual_request(self, client, mock_gmail_service, temp_subscribers_file):
        """Test CORS headers in actual request"""
        valid_data = {"email": "cors@example.com"}
        headers = {"Origin": "http://localhost:3000"}
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = client.post("/newsletter/signup", json=valid_data, headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]
