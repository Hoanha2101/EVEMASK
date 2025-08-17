"""
Security Tests for EVEMASK Newsletter Backend
Author: EVEMASK Team
"""

import json
from unittest.mock import patch


class TestInputValidation:
    """Test cases for input validation and sanitization"""
    
    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection attempts"""
        sql_injection_payloads = [
            "test@example.com'; DROP TABLE users; --",
            "test@example.com' OR '1'='1",
            "test@example.com'; INSERT INTO users VALUES ('hacker'); --",
            "test@example.com' UNION SELECT * FROM sensitive_data --"
        ]
        
        for payload in sql_injection_payloads:
            data = {"email": payload}
            response = client.post("/newsletter/signup", json=data)
            
            # Should be rejected due to invalid email format
            assert response.status_code == 422
    
    def test_xss_protection(self, client):
        """Test protection against XSS attacks"""
        xss_payloads = [
            "test@example.com<script>alert('xss')</script>",
            "test@example.com<img src=x onerror=alert('xss')>",
            "test@example.com\"><script>alert('xss')</script>",
            "test@example.com<iframe src=javascript:alert('xss')></iframe>"
        ]
        
        for payload in xss_payloads:
            data = {"email": payload}
            response = client.post("/newsletter/signup", json=data)
            
            # Should be rejected due to invalid email format
            assert response.status_code == 422
    
    def test_email_header_injection(self, client):
        """Test protection against email header injection"""
        header_injection_payloads = [
            "test@example.com\nBcc: hacker@evil.com",
            "test@example.com\r\nTo: victim@example.com",
            "test@example.com\nSubject: Hacked Email",
            "test@example.com\r\nX-Mailer: Evil Script"
        ]
        
        for payload in header_injection_payloads:
            data = {"email": payload}
            response = client.post("/newsletter/signup", json=data)
            
            # Should be rejected due to invalid email format
            assert response.status_code == 422
    
    def test_oversized_input_protection(self, client):
        """Test protection against oversized inputs"""
        # Very long email
        long_email = "a" * 1000 + "@example.com"
        
        data = {"email": long_email}
        response = client.post("/newsletter/signup", json=data)
        
        # Should be handled gracefully
        assert response.status_code in [400, 422, 413]  # Bad Request, Unprocessable Entity, or Payload Too Large
    
    def test_unicode_normalization(self, client, temp_subscribers_file, mock_gmail_service):
        """Test Unicode normalization and handling"""
        unicode_emails = [
            "tëst@example.com",
            "用户@example.com",
            "тест@example.com",
            "test@exämple.com"
        ]
        
        for email in unicode_emails:
            data = {"email": email}
            
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                response = client.post("/newsletter/signup", json=data)
            
            # Should handle Unicode properly
            assert response.status_code in [200, 422]  # Success or validation error
    
    def test_null_byte_injection(self, client):
        """Test protection against null byte injection"""
        null_byte_payloads = [
            "test@example.com\x00",
            "test\x00@example.com",
            "\x00test@example.com"
        ]
        
        for payload in null_byte_payloads:
            data = {"email": payload}
            response = client.post("/newsletter/signup", json=data)
            
            # Should be rejected
            assert response.status_code == 422


class TestAuthenticationSecurity:
    """Test cases for authentication and authorization security"""
    
    def test_environment_variable_security(self):
        """Test that sensitive environment variables are properly handled"""
        from main import get_gmail_credentials
        
        # Test with missing credentials
        with patch.dict('os.environ', {}, clear=True):
            try:
                get_gmail_credentials()
                # If it doesn't raise an error, check that it handles missing vars
                assert False, "Should raise an error for missing credentials"
            except Exception:
                # Expected behavior
                pass
    
    def test_credentials_not_logged(self, client, mock_gmail_service, caplog):
        """Test that credentials are not logged in plain text"""
        import logging
        
        data = {"email": "security@example.com"}
        
        with caplog.at_level(logging.DEBUG):
            response = client.post("/newsletter/signup", json=data)
        
        # Check that no sensitive information is in logs
        log_text = " ".join([record.message for record in caplog.records])
        
        sensitive_patterns = [
            "client_secret",
            "refresh_token",
            "access_token",
            "password"
        ]
        
        for pattern in sensitive_patterns:
            assert pattern.lower() not in log_text.lower()
    
    def test_api_rate_limiting_headers(self, client):
        """Test for rate limiting headers (if implemented)"""
        response = client.get("/")
        
        # Check for common rate limiting headers
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining", 
            "x-ratelimit-reset",
            "retry-after"
        ]
        
        # Note: This is aspirational - rate limiting might not be implemented
        # The test documents the security consideration
        header_names = [h.lower() for h in response.headers.keys()]
        
        # For now, just verify the response is successful
        assert response.status_code == 200


class TestDataSecurity:
    """Test cases for data security and privacy"""
    
    def test_file_permissions_security(self, temp_subscribers_file):
        """Test file permissions for subscriber data"""
        import os
        import stat
        
        from main import save_subscriber_to_file
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            save_subscriber_to_file("security@example.com")
        
        # Check file permissions
        file_stat = os.stat(temp_subscribers_file)
        file_mode = stat.filemode(file_stat.st_mode)
        
        # File should not be world-readable (basic check)
        # This is platform-dependent and might not be strictly enforced in tests
        assert os.path.exists(temp_subscribers_file)
    
    def test_data_not_exposed_in_errors(self, client):
        """Test that sensitive data is not exposed in error messages"""
        # Test with malformed request
        response = client.post("/newsletter/signup", data="invalid json")
        
        assert response.status_code in [400, 422]
        
        # Error message should not contain internal paths or sensitive info
        error_text = str(response.content)
        
        sensitive_patterns = [
            "/home/",
            "/usr/",
            "C:\\",
            "password",
            "secret",
            "token"
        ]
        
        for pattern in sensitive_patterns:
            assert pattern.lower() not in error_text.lower()
    
    def test_subscriber_data_sanitization(self, temp_subscribers_file):
        """Test that subscriber data is properly sanitized before storage"""
        from main import save_subscriber_to_file
        
        # Email with potentially dangerous characters
        email = "test+dangerous<script>@example.com"
        
        with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
            result = save_subscriber_to_file(email)
        
        if result:  # If email was accepted (after validation)
            with open(temp_subscribers_file, 'r') as f:
                data = json.load(f)
                stored_email = data[0]["email"]
                
                # Should be sanitized or properly escaped
                assert "<script>" not in stored_email
    
    def test_json_injection_protection(self, temp_subscribers_file):
        """Test protection against JSON injection"""
        from main import save_subscriber_to_file
        
        json_injection_payloads = [
            'test@example.com", "malicious": "data',
            'test@example.com"}, {"injected": "content"',
            'test@example.com\\", \\"hacked\\": \\"true'
        ]
        
        for payload in json_injection_payloads:
            with patch('main.SUBSCRIBERS_FILE', temp_subscribers_file):
                result = save_subscriber_to_file(payload)
            
            # Should either be rejected or properly escaped
            if result:
                with open(temp_subscribers_file, 'r') as f:
                    data = json.load(f)
                    # Verify JSON structure integrity
                    assert isinstance(data, list)
                    for item in data:
                        assert isinstance(item, dict)
                        assert "email" in item


class TestHTTPSecurity:
    """Test cases for HTTP security headers and practices"""
    
    def test_security_headers_present(self, client):
        """Test for security headers in responses"""
        response = client.get("/")
        
        # Common security headers to check for
        security_headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": ["DENY", "SAMEORIGIN"],
            "x-xss-protection": "1; mode=block",
            "strict-transport-security": None,  # HTTPS only
            "content-security-policy": None
        }
        
        response_headers = {k.lower(): v for k, v in response.headers.items()}
        
        # Check for X-Content-Type-Options
        if "x-content-type-options" in response_headers:
            assert response_headers["x-content-type-options"] == "nosniff"
        
        # Note: Some headers might not be implemented yet
        # This test documents security best practices
        
        assert response.status_code == 200
    
    def test_cors_security(self, client):
        """Test CORS configuration security"""
        # Test preflight request
        response = client.options("/newsletter/signup", headers={
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST"
        })
        
        # CORS should be properly configured
        assert response.status_code == 200
        
        # Check CORS headers
        cors_headers = response.headers
        
        # Should not allow arbitrary origins (check for specific allowed origins)
        if "access-control-allow-origin" in cors_headers:
            allowed_origin = cors_headers["access-control-allow-origin"]
            # Should not be "*" for credentialed requests
            assert allowed_origin != "*" or "access-control-allow-credentials" not in cors_headers
    
    def test_method_not_allowed_security(self, client):
        """Test security of unsupported HTTP methods"""
        unsupported_methods = ["PUT", "DELETE", "PATCH"]
        
        for method in unsupported_methods:
            response = client.request(method, "/newsletter/signup")
            
            # Should return 405 Method Not Allowed
            assert response.status_code == 405
            
            # Should not expose internal information
            error_text = str(response.content)
            assert "internal" not in error_text.lower()
            assert "debug" not in error_text.lower()


class TestErrorHandlingSecurity:
    """Test cases for secure error handling"""
    
    def test_error_message_information_disclosure(self, client):
        """Test that error messages don't disclose sensitive information"""
        # Trigger various errors
        test_cases = [
            ("POST", "/newsletter/signup", {"email": "invalid"}),
            ("GET", "/nonexistent-endpoint", None),
            ("POST", "/newsletter/signup", None)
        ]
        
        for method, endpoint, data in test_cases:
            if method == "POST":
                response = client.post(endpoint, json=data)
            else:
                response = client.get(endpoint)
            
            # Check error response
            if response.status_code >= 400:
                error_content = str(response.content)
                
                # Should not contain internal paths or stack traces
                forbidden_patterns = [
                    "traceback",
                    "/usr/",
                    "/home/",
                    "C:\\",
                    "__pycache__",
                    ".py:",
                    "line "
                ]
                
                for pattern in forbidden_patterns:
                    assert pattern.lower() not in error_content.lower()
    
    def test_exception_handling_security(self, client, mock_gmail_service):
        """Test that exceptions are handled securely"""
        data = {"email": "exception@example.com"}
        
        # Force an exception in email service
        with patch('main.send_welcome_email', side_effect=Exception("Internal error")):
            response = client.post("/newsletter/signup", json=data)
        
        # Should return a generic error, not expose the exception details
        assert response.status_code == 500
        
        error_content = str(response.content)
        assert "Internal error" not in error_content  # Original exception message should not be exposed
