"""
Email Service Tests for EVEMASK Newsletter Backend
Author: EVEMASK Team
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from google.auth.credentials import Credentials


class TestEmailService:
    """Test cases for email service functionality"""
    
    def test_send_welcome_email_success(self, mock_gmail_service, mock_gmail_credentials):
        """Test successful welcome email sending"""
        from main import send_welcome_email
        
        # Mock the service build and message sending
        with patch('main.build', return_value=mock_gmail_service):
            result = send_welcome_email("test@example.com")
        
        assert result is True
        mock_gmail_service.users().messages().send.assert_called_once()
    
    def test_send_welcome_email_gmail_api_error(self, mock_gmail_credentials):
        """Test email sending with Gmail API error"""
        from main import send_welcome_email
        
        with patch('main.build', side_effect=Exception("Gmail API connection failed")):
            result = send_welcome_email("test@example.com")
        
        assert result is False
    
    def test_send_welcome_email_invalid_credentials(self):
        """Test email sending with invalid credentials"""
        from main import send_welcome_email
        
        with patch('main.Credentials.from_authorized_user_info', side_effect=Exception("Invalid credentials")):
            result = send_welcome_email("test@example.com")
        
        assert result is False
    
    def test_email_template_generation(self):
        """Test HTML email template generation"""
        from main import create_welcome_email_html
        
        email = "test@example.com"
        html_content = create_welcome_email_html(email)
        
        assert isinstance(html_content, str)
        assert "Welcome to EVEMASK" in html_content
        assert email in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "</html>" in html_content
    
    def test_email_template_special_characters(self):
        """Test email template with special characters in email"""
        from main import create_welcome_email_html
        
        email = "test+special@example.com"
        html_content = create_welcome_email_html(email)
        
        assert email in html_content
        assert "Welcome to EVEMASK" in html_content
    
    def test_credentials_loading_success(self):
        """Test successful credentials loading from environment"""
        from main import get_gmail_credentials
        
        mock_creds_data = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "refresh_token": "test_refresh_token",
            "type": "authorized_user"
        }
        
        with patch('main.Credentials.from_authorized_user_info') as mock_creds:
            mock_credentials = Mock()
            mock_creds.return_value = mock_credentials
            
            result = get_gmail_credentials()
            
            assert result == mock_credentials
            mock_creds.assert_called_once()
    
    def test_credentials_loading_missing_env_vars(self):
        """Test credentials loading with missing environment variables"""
        from main import get_gmail_credentials
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(Exception):
                get_gmail_credentials()
    
    def test_email_message_format(self, mock_gmail_service, mock_gmail_credentials):
        """Test email message format and structure"""
        from main import send_welcome_email
        
        test_email = "format@example.com"
        
        with patch('main.build', return_value=mock_gmail_service):
            send_welcome_email(test_email)
        
        # Get the call arguments
        call_args = mock_gmail_service.users().messages().send.call_args
        
        # Verify message structure exists
        assert call_args is not None
    
    def test_email_encoding_utf8(self):
        """Test email content properly handles UTF-8 encoding"""
        from main import create_welcome_email_html
        
        email_with_unicode = "tést@éxample.com"
        html_content = create_welcome_email_html(email_with_unicode)
        
        assert email_with_unicode in html_content
        # Verify it can be encoded/decoded
        encoded = html_content.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert decoded == html_content


class TestEmailValidation:
    """Test cases for email validation functionality"""
    
    def test_valid_email_formats(self):
        """Test various valid email formats"""
        from main import NewsletterSignup
        
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user123@example-domain.com",
            "firstname.lastname@subdomain.example.com"
        ]
        
        for email in valid_emails:
            # Should not raise validation error
            signup = NewsletterSignup(email=email)
            assert signup.email == email
    
    def test_invalid_email_formats(self):
        """Test various invalid email formats"""
        from main import NewsletterSignup
        from pydantic import ValidationError
        
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "user@",
            "user.example.com",
            "user@.com",
            "user@@example.com",
            "",
            "user @example.com",
            "user@exam ple.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                NewsletterSignup(email=email)
    
    def test_email_normalization(self):
        """Test email normalization (lowercase conversion)"""
        from main import NewsletterSignup
        
        mixed_case_email = "User@Example.COM"
        signup = NewsletterSignup(email=mixed_case_email)
        
        # Should be normalized to lowercase
        assert signup.email == "user@example.com"


class TestEmailRateLimiting:
    """Test cases for email rate limiting (if implemented)"""
    
    def test_multiple_rapid_emails_same_address(self, client, temp_subscribers_file, mock_gmail_service):
        """Test sending multiple emails to same address rapidly"""
        email_data = {"email": "rapid@example.com"}
        
        # First signup
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response1 = client.post("/newsletter/signup", json=email_data)
        
        # Immediate second signup (should be blocked due to duplicate)
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            response2 = client.post("/newsletter/signup", json=email_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 409  # Conflict for duplicate
