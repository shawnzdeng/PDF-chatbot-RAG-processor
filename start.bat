@echo off
echo PDF RAG System - Quick Start
echo ============================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo Error: .env file not found
    echo Please create .env file with your API keys
    echo.
    echo Required format:
    echo OPENAI_API_KEY=your_key_here
    echo QDRANT_API_KEY=your_key_here  
    echo QDRANT_URL=your_qdrant_url_here
    pause
    exit /b 1
)

REM Install requirements if not already installed
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Choose an option:
echo 1. Run quick test
echo 2. Process PDF
echo 3. Run demo
echo 4. Start parameter tuning
echo 5. Interactive query mode
echo 6. View MLflow UI
echo.

set /p choice=Enter choice (1-6): 

if "%choice%"=="1" (
    python setup.py --test
) else if "%choice%"=="2" (
    python main.py process
) else if "%choice%"=="3" (
    python setup.py --demo
) else if "%choice%"=="4" (
    python main.py tune --max-combinations 18
) else if "%choice%"=="5" (
    python main.py query --interactive
) else if "%choice%"=="6" (
    mlflow ui --backend-store-uri file:./mlruns
) else (
    echo Invalid choice
)

pause
