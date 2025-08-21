# EVEMASK Backend Testing Suite

## Overview
Comprehensive testing framework for the EVEMASK Newsletter API backend service. This test suite ensures reliability, performance, security, and functionality of the newsletter signup and email service components.

## Test Coverage Summary

### 📊 Test Statistics
- **Total Test Files**: 7
- **Estimated Test Cases**: 150+
- **Test Categories**: 6
- **Coverage Target**: 80%+

## Test Structure

### 🧪 Test Categories

#### 1. Unit Tests (`test_api_endpoints.py`)
- **Purpose**: Test individual API endpoints in isolation
- **Test Count**: ~25 tests
- **Coverage**:
  - Newsletter signup endpoint validation
  - Email format validation and sanitization
  - Response format verification
  - Error handling for invalid inputs
  - CORS headers validation
  - Duplicate email detection

#### 2. Email Service Tests (`test_email_service.py`)
- **Purpose**: Test email functionality and Gmail API integration
- **Test Count**: ~20 tests
- **Coverage**:
  - Gmail API authentication and service building
  - Welcome email template generation
  - Email sending success/failure scenarios
  - Credential loading and validation
  - HTML email content verification
  - Unicode and special character handling

#### 3. Data Persistence Tests (`test_data_persistence.py`)
- **Purpose**: Test data storage and retrieval functionality
- **Test Count**: ~30 tests  
- **Coverage**:
  - JSON file creation and management
  - Subscriber data structure validation
  - File permission and access handling
  - Data integrity and consistency checks
  - Concurrent access protection
  - Large dataset handling
  - UTF-8 encoding support

#### 4. Performance Tests (`test_performance.py`)
- **Purpose**: Test system performance under various load conditions
- **Test Count**: ~25 tests
- **Coverage**:
  - Response time measurement
  - Concurrent request handling
  - Memory usage monitoring
  - File I/O performance
  - Load testing scenarios
  - Resource utilization tracking

#### 5. Security Tests (`test_security.py`)
- **Purpose**: Test security measures and vulnerability protection
- **Test Count**: ~35 tests
- **Coverage**:
  - Input validation and sanitization
  - SQL injection protection
  - XSS attack prevention
  - Email header injection protection
  - Authentication security
  - Data exposure prevention
  - HTTP security headers
  - Error message security

#### 6. Integration Tests (`test_integration.py`)
- **Purpose**: Test end-to-end workflows and component integration
- **Test Count**: ~25 tests
- **Coverage**:
  - Complete signup workflow
  - Multi-user scenarios
  - Service integration testing
  - Error handling integration
  - Configuration integration
  - Async operation testing

### 🔧 Test Configuration

#### Framework Setup
- **Primary Framework**: pytest
- **Async Support**: pytest-asyncio
- **Mocking**: unittest.mock + pytest-mock
- **Coverage**: pytest-cov
- **Performance**: psutil for resource monitoring

#### Test Fixtures (`conftest.py`)
- FastAPI TestClient setup
- Async client configuration  
- Mock Gmail service
- Temporary file management
- Environment variable mocking
- Sample data generation

## Running Tests

### Quick Start
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run all tests
python run_tests.py

# Run specific test category
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type performance
python run_tests.py --type security
```

### Advanced Usage
```bash
# Run with coverage report
python run_tests.py --type all

# Run fast tests only
python run_tests.py --type fast

# Run with linting
python run_tests.py --lint

# Run with security checks
python run_tests.py --security

# Skip coverage reporting
python run_tests.py --no-coverage
```

### Direct pytest Usage
```bash
# Run specific test file
pytest tests/test_api_endpoints.py -v

# Run with markers
pytest -m "unit" -v
pytest -m "security and not slow" -v

# Run with coverage
pytest --cov=main --cov-report=html tests/
```

## Test Environment

### Required Dependencies
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.24.0
psutil>=5.9.0
pytest-cov>=4.1.0
```

### Environment Variables
- `TESTING=true` - Enables test mode
- `ENVIRONMENT=test` - Sets test environment
- Mock Gmail credentials automatically provided

### Test Data
- Temporary JSON files for subscriber storage
- Mock Gmail API responses
- Sample email templates
- Test user datasets

## Coverage Goals

### Target Metrics
- **Line Coverage**: ≥80%
- **Function Coverage**: ≥90%
- **Branch Coverage**: ≥75%

### Critical Areas
- All API endpoints
- Email service functions
- Data persistence operations
- Error handling paths
- Security validation logic

## Test Reports

### Generated Reports
- **HTML Coverage**: `htmlcov/index.html`
- **HTML Test Report**: `reports/report.html`
- **JSON Report**: `reports/report.json`
- **XML Coverage**: `coverage.xml`

### CI/CD Integration
- JUnit XML format support
- Coverage reporting for CI systems
- Test result artifacts
- Performance benchmarking

## Security Testing

### Vulnerability Checks
- Input validation testing
- Injection attack prevention
- Authentication security
- Data exposure protection
- HTTP security headers

### Tools Integration
- Bandit for security analysis
- Safety for dependency checks
- Custom security test scenarios

## Performance Benchmarks

### Response Time Targets
- Newsletter signup: <2 seconds
- Subscriber retrieval: <1 second
- Health check: <0.5 seconds

### Load Testing
- Concurrent request handling
- Memory usage monitoring
- File I/O performance
- Email service performance

## Best Practices

### Test Writing Guidelines
1. Use descriptive test names
2. Follow AAA pattern (Arrange, Act, Assert)
3. Mock external dependencies
4. Test both success and failure paths
5. Include edge case scenarios

### Maintenance
- Regular test dependency updates
- Performance benchmark reviews
- Security test updates
- Coverage goal adjustments

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all test dependencies are installed
2. **File Permissions**: Check temporary file creation permissions
3. **Mock Failures**: Verify mock configurations match actual implementations
4. **Performance Failures**: Adjust timeout values for slower systems

### Debug Mode
```bash
# Run with verbose output
pytest -v -s tests/

# Run single test with debugging
pytest tests/test_api_endpoints.py::TestNewsletterSignupAPI::test_newsletter_signup_success -v -s
```

## Contributing

### Adding New Tests
1. Follow existing test structure
2. Add appropriate markers
3. Update documentation
4. Ensure coverage targets are met

### Test Categories
Use pytest markers to categorize tests:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Long-running tests

---

**Author**: EVEMASK Team  
**Last Updated**: January 2025  
**Version**: 1.0.0
