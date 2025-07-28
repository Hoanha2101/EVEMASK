@echo off
echo Starting EVEMASK Newsletter API Server...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if main.py exists
if not exist main.py (
    echo Error: main.py not found!
    echo Please make sure you're in the correct directory.
    pause
    exit /b 1
)

REM Start the FastAPI server
echo Server starting at http://localhost:8002
echo API Documentation available at http://localhost:8002/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py
