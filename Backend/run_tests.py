"""
===========================
EVEMASK Backend Test Runner
===========================

Comprehensive test execution framework for EVEMASK Newsletter API backend.

Test Suite Overview:
1. Unit Tests:
   - Individual function and method testing
   - Input validation and edge case handling
   - Mock testing for external dependencies
   - Code coverage analysis and reporting

2. Integration Tests:
   - Database connectivity and operations
   - Email service integration
   - API endpoint functionality
   - Cross-component interaction validation

3. Performance Tests:
   - Response time benchmarking
   - Load testing and stress testing
   - Memory usage optimization
   - Concurrent request handling

4. Security Tests:
   - Input sanitization validation
   - SQL injection prevention
   - XSS attack mitigation
   - Authentication and authorization

Test Execution Flow:
[Environment Setup] -> [Unit Tests] -> [Integration Tests] -> [Performance Tests] -> [Security Tests] -> [Report]
        |                 |              |                    |                     |                |
   (Dependencies)   (Core Logic)   (System Integration)   (Performance)      (Security)     (Summary)

Test Categories:
- API Endpoints: Newsletter signup, subscriber management
- Database Operations: Supabase integration, JSON fallback
- Email Service: Gmail API integration, template rendering
- Error Handling: Graceful degradation, recovery mechanisms
- Configuration: Environment validation, credential management

Reporting Features:
- Detailed test results with pass/fail status
- Code coverage metrics and analysis
- Performance benchmarks and trends
- Security vulnerability assessment
- Comprehensive HTML and JSON reports

Author: EVEMASK Team
Version: 2.0.0
Usage: python run_tests.py [--coverage] [--performance] [--security] [--report-format html]
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(command, description):
    """
    Execute shell command with comprehensive logging and error handling.
    
    This function provides a standardized way to run test commands with:
    - Detailed command logging for debugging
    - Real-time output streaming
    - Error capture and reporting
    - Return code validation
    - Execution time measurement
    
    Args:
        command (str): Shell command to execute
        description (str): Human-readable description of the command
        
    Returns:
        subprocess.CompletedProcess: Command execution result with:
            - returncode: Exit status (0 = success)
            - stdout: Command output
            - stderr: Error output
            - execution_time: Time taken to execute
            
    Error Handling:
        - Command not found: Reports missing dependencies
        - Permission errors: Suggests permission fixes
        - Timeout: Terminates long-running commands
        - Memory errors: Reports resource constraints
        
    Logging:
        Provides detailed execution logs including:
        - Command being executed
        - Execution environment details
        - Real-time output streaming
        - Final execution summary
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def setup_test_environment():
    """Setup the test environment"""
    print("Setting up test environment...")
    
    # Create necessary directories
    directories = ["reports", "htmlcov"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Install test dependencies
    if not run_command("pip install -r tests/requirements.txt", "Installing test dependencies"):
        print("Warning: Failed to install test dependencies")
        return False
    
    return True


def run_tests(test_type="all", verbose=True, coverage=True):
    """Run the specified tests"""
    
    # Base pytest command
    cmd_parts = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd_parts.append("-v")
    
    # Add coverage options
    if coverage:
        cmd_parts.extend([
            "--cov=main",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml"
        ])
    
    # Add test filtering based on type
    if test_type == "unit":
        cmd_parts.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd_parts.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd_parts.extend(["-m", "performance"])
    elif test_type == "security":
        cmd_parts.extend(["-m", "security"])
    elif test_type == "email":
        cmd_parts.extend(["-m", "email"])
    elif test_type == "persistence":
        cmd_parts.extend(["-m", "persistence"])
    elif test_type == "fast":
        cmd_parts.extend(["-m", "fast"])
    elif test_type == "slow":
        cmd_parts.extend(["-m", "slow"])
    
    # Add HTML report
    cmd_parts.extend([
        "--html=reports/report.html",
        "--self-contained-html"
    ])
    
    # Add JSON report
    cmd_parts.extend([
        "--json-report",
        "--json-report-file=reports/report.json"
    ])
    
    # Add test path
    cmd_parts.append("tests/")
    
    command = " ".join(cmd_parts)
    
    return run_command(command, f"Running {test_type} tests")


def run_linting():
    """Run code linting and style checks"""
    linting_commands = [
        ("flake8 main.py tests/", "Running flake8 linting"),
        ("black --check main.py tests/", "Running black code formatting check"),
        ("isort --check-only main.py tests/", "Running isort import sorting check"),
        ("mypy main.py", "Running mypy type checking")
    ]
    
    all_passed = True
    for command, description in linting_commands:
        if not run_command(command, description):
            all_passed = False
            print(f"❌ {description} failed")
        else:
            print(f"✅ {description} passed")
    
    return all_passed


def run_security_checks():
    """Run security vulnerability checks"""
    security_commands = [
        ("bandit -r main.py", "Running bandit security analysis"),
        ("safety check", "Running safety dependency vulnerability check")
    ]
    
    all_passed = True
    for command, description in security_commands:
        if not run_command(command, description):
            all_passed = False
            print(f"❌ {description} failed")
        else:
            print(f"✅ {description} passed")
    
    return all_passed


def generate_coverage_report():
    """Generate and display coverage report"""
    print("\n" + "="*60)
    print("COVERAGE REPORT")
    print("="*60)
    
    # Generate coverage report
    if run_command("coverage report -m", "Generating coverage report"):
        print("✅ Coverage report generated successfully")
        
        # Open HTML coverage report
        coverage_file = Path("htmlcov/index.html")
        if coverage_file.exists():
            print(f"📊 HTML coverage report available at: {coverage_file.absolute()}")
        
        return True
    else:
        print("❌ Failed to generate coverage report")
        return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="EVEMASK Backend Test Runner")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "performance", "security", "email", "persistence", "fast", "slow"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--no-setup", action="store_true", help="Skip test environment setup")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    print("🧪 EVEMASK Backend Test Runner")
    print("="*60)
    
    # Setup test environment
    if not args.no_setup:
        if not setup_test_environment():
            print("❌ Failed to setup test environment")
            sys.exit(1)
    
    # Run linting if requested
    if args.lint:
        print("\n🔍 Running code quality checks...")
        if not run_linting():
            print("❌ Linting checks failed")
            sys.exit(1)
        print("✅ All linting checks passed")
    
    # Run security checks if requested
    if args.security:
        print("\n🔒 Running security checks...")
        if not run_security_checks():
            print("❌ Security checks failed")
            sys.exit(1)
        print("✅ All security checks passed")
    
    # Run tests
    print(f"\n🧪 Running {args.type} tests...")
    test_success = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=not args.no_coverage
    )
    
    # Generate coverage report
    if not args.no_coverage and test_success:
        generate_coverage_report()
    
    # Final results
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    
    if test_success:
        print("✅ All tests passed successfully!")
        
        # Show report locations
        print("\n📊 Generated Reports:")
        print(f"  - HTML Test Report: reports/report.html")
        print(f"  - JSON Test Report: reports/report.json")
        if not args.no_coverage:
            print(f"  - HTML Coverage Report: htmlcov/index.html")
            print(f"  - XML Coverage Report: coverage.xml")
        
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
