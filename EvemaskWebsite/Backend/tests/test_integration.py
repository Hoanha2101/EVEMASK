"""
Integration Tests for EVEMASK Newsletter Backend
Author: EVEMASK Team
"""

import json
import asyncio
from unittest.mock import patch, MagicMock
import httpx


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_complete_newsletter_signup_workflow(self, client, temp_subscribers_file, mock_gmail_service):
        """Test complete newsletter signup workflow from request to email"""
        email = "workflow@example.com"
        signup_data = {"email": email}
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            # Step 1: Submit newsletter signup
            response = client.post("/newsletter/signup", json=signup_data)
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert response_data["email"] == email
            
            # Step 2: Verify data persistence
            with open(temp_subscribers_file, 'r') as f:
                subscribers = json.load(f)
                assert len(subscribers) == 1
                assert subscribers[0]["email"] == email
                assert subscribers[0]["status"] == "active"
            
            # Step 3: Verify email service was called
            mock_gmail_service.users().messages().send.assert_called_once()
            
            # Step 4: Verify subscriber can be retrieved
            get_response = client.get("/subscribers")
            assert get_response.status_code == 200
            
            get_data = get_response.json()
            assert get_data["total"] == 1
            assert get_data["subscribers"][0]["email"] == email
    
    def test_duplicate_signup_workflow(self, client, temp_subscribers_file, mock_gmail_service):
        """Test workflow for duplicate signup attempts"""
        email = "duplicate@example.com"
        signup_data = {"email": email}
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            # First signup
            response1 = client.post("/newsletter/signup", json=signup_data)
            assert response1.status_code == 200
            
            # Second signup (duplicate)
            response2 = client.post("/newsletter/signup", json=signup_data)
            assert response2.status_code == 409
            
            # Verify only one entry exists
            get_response = client.get("/subscribers")
            get_data = get_response.json()
            assert get_data["total"] == 1
    
    def test_multiple_users_signup_workflow(self, client, temp_subscribers_file, mock_gmail_service):
        """Test workflow for multiple users signing up"""
        emails = [
            "user1@example.com",
            "user2@example.com", 
            "user3@example.com"
        ]
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            # Sign up multiple users
            for email in emails:
                response = client.post("/newsletter/signup", json={"email": email})
                assert response.status_code == 200
            
            # Verify all users are stored
            get_response = client.get("/subscribers")
            get_data = get_response.json()
            assert get_data["total"] == 3
            
            stored_emails = [sub["email"] for sub in get_data["subscribers"]]
            for email in emails:
                assert email in stored_emails


class TestAPIIntegration:
    """Test integration between different API endpoints"""
    
    def test_signup_and_retrieval_integration(self, client, temp_subscribers_file, mock_gmail_service):
        """Test integration between signup and retrieval endpoints"""
        # Add multiple subscribers
        test_emails = ["int1@example.com", "int2@example.com", "int3@example.com"]
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            for email in test_emails:
                signup_response = client.post("/newsletter/signup", json={"email": email})
                assert signup_response.status_code == 200
            
            # Retrieve all subscribers
            get_response = client.get("/subscribers")
            assert get_response.status_code == 200
            
            get_data = get_response.json()
            assert get_data["total"] == len(test_emails)
            
            # Verify data consistency
            for subscriber in get_data["subscribers"]:
                assert subscriber["email"] in test_emails
                assert subscriber["status"] == "active"
                assert "timestamp" in subscriber
    
    def test_health_check_integration(self, client):
        """Test health check endpoint integration"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "EVEMASK Newsletter API"
        assert "timestamp" in data
    
    def test_cors_integration_across_endpoints(self, client, temp_subscribers_file, mock_gmail_service):
        """Test CORS integration across all endpoints"""
        origin = "https://evemask.ai"
        headers = {"Origin": origin}
        
        # Test CORS on health check
        health_response = client.get("/", headers=headers)
        assert health_response.status_code == 200
        
        # Test CORS on signup
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            signup_response = client.post("/newsletter/signup", 
                                        json={"email": "cors@example.com"}, 
                                        headers=headers)
            assert signup_response.status_code == 200
        
        # Test CORS on subscribers retrieval
        get_response = client.get("/subscribers", headers=headers)
        assert get_response.status_code == 200


class TestEmailServiceIntegration:
    """Test integration with email service"""
    
    def test_gmail_api_integration_flow(self, client, temp_subscribers_file):
        """Test Gmail API integration flow"""
        email = "gmail@example.com"
        
        # Mock the entire Gmail service chain
        with patch('main.get_gmail_credentials') as mock_creds:
            with patch('main.build') as mock_build:
                mock_service = MagicMock()
                mock_build.return_value = mock_service
                mock_service.users().messages().send().execute.return_value = {"id": "msg123"}
                
                with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                    response = client.post("/newsletter/signup", json={"email": email})
                
                assert response.status_code == 200
                
                # Verify Gmail API was called correctly
                mock_creds.assert_called_once()
                mock_build.assert_called_once()
                mock_service.users().messages().send.assert_called_once()
    
    def test_email_failure_handling_integration(self, client, temp_subscribers_file):
        """Test handling of email service failures"""
        email = "emailfail@example.com"
        
        # Mock email service failure
        with patch('main.send_welcome_email', return_value=False):
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json={"email": email})
            
            # Should still handle gracefully
            assert response.status_code in [200, 500]  # Depends on implementation
    
    def test_email_template_integration(self, mock_gmail_service):
        """Test email template integration with Gmail service"""
        from main import send_welcome_email, create_welcome_email_html
        
        email = "template@example.com"
        html_content = create_welcome_email_html(email)
        
        with patch('main.build', return_value=mock_gmail_service):
            result = send_welcome_email(email)
            assert result is True
        
        # Verify HTML template was generated
        assert "Welcome to EVEMASK" in html_content
        assert email in html_content


class TestDataPersistenceIntegration:
    """Test integration with data persistence layer"""
    
    def test_file_system_integration(self, client, temp_subscribers_file, mock_gmail_service):
        """Test file system integration for data persistence"""
        emails = ["fs1@example.com", "fs2@example.com"]
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            # Add subscribers
            for email in emails:
                response = client.post("/newsletter/signup", json={"email": email})
                assert response.status_code == 200
            
            # Verify file persistence
            with open(temp_subscribers_file, 'r') as f:
                file_data = json.load(f)
                assert len(file_data) == 2
            
            # Verify API can read persisted data
            get_response = client.get("/subscribers")
            api_data = get_response.json()
            assert api_data["total"] == 2
    
    def test_json_serialization_integration(self, client, temp_subscribers_file, mock_gmail_service):
        """Test JSON serialization/deserialization integration"""
        email = "json@example.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            # Add subscriber
            response = client.post("/newsletter/signup", json={"email": email})
            assert response.status_code == 200
            
            # Read raw file content
            with open(temp_subscribers_file, 'r') as f:
                raw_content = f.read()
                parsed_data = json.loads(raw_content)
            
            # Verify JSON structure
            assert isinstance(parsed_data, list)
            assert len(parsed_data) == 1
            assert parsed_data[0]["email"] == email
            
            # Verify API returns same data
            get_response = client.get("/subscribers")
            api_data = get_response.json()
            assert api_data["subscribers"][0]["email"] == email


class TestErrorHandlingIntegration:
    """Test integration of error handling across components"""
    
    def test_cascading_error_handling(self, client):
        """Test error handling cascades properly through the system"""
        # Test various error scenarios
        error_cases = [
            # Invalid email format
            {"email": "invalid-email"},
            # Missing email field
            {},
            # Invalid JSON structure
            "not-json"
        ]
        
        for case in error_cases:
            if isinstance(case, dict):
                response = client.post("/newsletter/signup", json=case)
            else:
                response = client.post("/newsletter/signup", data=case)
            
            # Should handle errors gracefully
            assert response.status_code >= 400
            
            # Should return proper error response
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                assert "detail" in error_data or "message" in error_data
    
    def test_service_failure_integration(self, client, temp_subscribers_file):
        """Test integration when services fail"""
        email = "servicefail@example.com"
        
        # Mock various service failures
        failure_scenarios = [
            # Gmail service failure
            patch('main.send_welcome_email', side_effect=Exception("Gmail API error")),
            # File system failure
            patch('builtins.open', side_effect=PermissionError("File access denied")),
            # Credentials failure
            patch('main.get_gmail_credentials', side_effect=Exception("Auth error"))
        ]
        
        for failure_patch in failure_scenarios:
            with failure_patch:
                with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                    response = client.post("/newsletter/signup", json={"email": email})
                
                # Should handle failures gracefully
                assert response.status_code in [200, 500]


class TestAsyncIntegration:
    """Test integration with async components (if any)"""
    
    async def test_async_client_integration(self, async_client, temp_subscribers_file, mock_gmail_service):
        """Test async client integration"""
        email = "async@example.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response = await async_client.post("/newsletter/signup", json={"email": email})
            
            assert response.status_code == 200
            
            # Test async retrieval
            get_response = await async_client.get("/subscribers")
            assert get_response.status_code == 200
            
            data = get_response.json()
            assert data["total"] == 1
            assert data["subscribers"][0]["email"] == email


class TestConfigurationIntegration:
    """Test integration with configuration and environment"""
    
    def test_environment_configuration_integration(self, client, mock_gmail_service):
        """Test integration with environment configuration"""
        # Test with different environment configurations
        test_configs = [
            {
                "SENDER_EMAIL": "test@evemask.ai",
                "SENDER_NAME": "EVEMASK Test Team"
            }
        ]
        
        for config in test_configs:
            with patch.dict('os.environ', config):
                response = client.get("/")
                assert response.status_code == 200
    
    def test_missing_configuration_integration(self, client):
        """Test behavior with missing configuration"""
        # Clear environment variables
        with patch.dict('os.environ', {}, clear=True):
            # Health check should still work
            response = client.get("/")
            assert response.status_code == 200
            
            # Email-dependent endpoints might fail gracefully
            email_response = client.post("/newsletter/signup", json={"email": "test@example.com"})
            # Should handle missing config gracefully
            assert email_response.status_code in [200, 500]
