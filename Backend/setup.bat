@echo off
echo Installing EVEMASK Newsletter API...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo SETUP COMPLETED!
echo ========================================
echo.
echo To run the API server:
echo 1. Run: start_server.bat
echo 2. Or manually: 
echo    - call venv\Scripts\activate.bat
echo    - python main.py
echo.
echo The API will be available at: http://localhost:8002
echo API Documentation: http://localhost:8002/docs
echo.

echo IMPORTANT: 
echo 1. Copy .env.example to .env
echo 2. Configure your Supabase credentials (SUPABASE_URL, SUPABASE_ANON_KEY)
echo 3. Configure your Gmail API credentials (follow OAUTH_GUIDE.md)
echo 4. Run migration script if you have existing JSON data: python migrate_to_supabase.py
echo For detailed setup instructions, see SUPABASE_SETUP.md
echo.
pause
