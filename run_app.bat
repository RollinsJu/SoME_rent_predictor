@echo off
setlocal EnableExtensions

REM Always run from the folder where this file lives.
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

cls
echo ============================================================
echo Southern Maine Rental Price Predictor
echo ============================================================
echo.
echo IMPORTANT: Run this file from the extracted app folder.
echo Do not run it directly from inside the ZIP preview window.
echo.
echo This launcher will set up a local Python environment if needed,
echo install the app requirements, and start the Streamlit app.
echo.
echo Keep this window open while using the app.
echo.

REM Check that the required files are present before creating/installing anything.
if not exist "%APP_DIR%requirements.txt" (
    echo The file requirements.txt was not found in this folder.
    echo.
    echo This usually means the app was run before the ZIP fully extracted,
    echo or it was opened directly from the compressed ZIP preview.
    echo.
    echo Please do this:
    echo   1. Right-click the ZIP file.
    echo   2. Choose Extract All.
    echo   3. Open the extracted some_rent_streamlit_app folder.
    echo   4. Double-click run_app.bat from that extracted folder.
    echo.
    echo Current folder:
    echo %APP_DIR%
    echo.
    pause
    exit /b 1
)

if not exist "%APP_DIR%app.py" (
    echo The file app.py was not found in this folder.
    echo Please make sure the ZIP was fully extracted before running this launcher.
    echo.
    pause
    exit /b 1
)

if not exist "%APP_DIR%model_utils.py" (
    echo The file model_utils.py was not found in this folder.
    echo Please make sure the ZIP was fully extracted before running this launcher.
    echo.
    pause
    exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=python"
    ) else (
        echo Python was not found on this computer.
        echo Please install Python 3 from https://www.python.org/downloads/
        echo Make sure to check "Add python.exe to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%APP_DIR%.venv\Scripts\python.exe" (
    echo Creating local app environment...
    %PYTHON_CMD% -m venv "%APP_DIR%.venv"
    if errorlevel 1 (
        echo Failed to create the local Python environment.
        pause
        exit /b 1
    )
)

call "%APP_DIR%.venv\Scripts\activate.bat"

echo Installing or checking required packages...
python -m pip install --upgrade pip
python -m pip install -r "%APP_DIR%requirements.txt"
if errorlevel 1 (
    echo.
    echo Package installation failed. Check the error message above.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit. A browser window should open automatically.
echo.
python -m streamlit run "%APP_DIR%app.py"

echo.
echo Streamlit has stopped.
pause
