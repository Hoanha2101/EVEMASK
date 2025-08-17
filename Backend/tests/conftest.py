"""
Pytest configuration and fixtures for EVEMASK Backend testing
Author: EVEMASK Team
"""

import pytest
import asyncio
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest_asyncio

# Import the main app
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import app

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create TestClient for FastAPI app"""
    return TestClient(app)

@pytest_asyncio.fixture
async def async_client():
    """Create async client for FastAPI app"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def temp_subscribers_file():
    """Create temporary subscribers file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([], f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    try:
        os.unlink(temp_file)
    except:
        pass

@pytest.fixture
def mock_gmail_service():
    """Mock Gmail API service"""
    with patch('main.build') as mock_build:
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Mock successful email sending
        mock_service.users().messages().send().execute.return_value = {"id": "test_message_id"}
        
        yield mock_service

@pytest.fixture
def mock_gmail_credentials():
    """Mock Gmail credentials"""
    with patch('main.Credentials.from_authorized_user_info') as mock_creds:
        mock_credentials = Mock()
        mock_creds.return_value = mock_credentials
        yield mock_credentials

@pytest.fixture
def valid_email_data():
    """Valid email data for testing"""
    return {
        "email": "test@example.com"
    }

@pytest.fixture
def invalid_email_data():
    """Invalid email data for testing"""
    return [
        {"email": "invalid-email"},
        {"email": ""},
        {"email": "test@"},
        {"email": "@example.com"},
        {"email": "test.example.com"},
        {}
    ]

@pytest.fixture
def sample_subscribers():
    """Sample subscribers data"""
    return [
        {
            "email": "user1@example.com",
            "timestamp": "2025-01-01T10:00:00",
            "status": "active"
        },
        {
            "email": "user2@example.com",
            "timestamp": "2025-01-01T11:00:00",
            "status": "active"
        }
    ]

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup environment variables for testing"""
    env_vars = {
        "SENDER_EMAIL": "test@evemask.ai",
        "GOOGLE_CLIENT_ID": "test_client_id",
        "GOOGLE_CLIENT_SECRET": "test_client_secret", 
        "GOOGLE_REFRESH_TOKEN": "test_refresh_token",
        "SENDER_NAME": "EVEMASK Test Team"
    }
    
    with patch.dict(os.environ, env_vars):
        yield
