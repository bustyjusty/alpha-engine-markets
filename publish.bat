@echo off
REM Publish to the live site. Pushing IS deploying: Streamlit Cloud rebuilds
REM from whatever lands on main, so the checks run BEFORE the push, not after.
REM
REM   publish.bat "Add the snapshot page"
REM
REM Quote the message. Unquoted, the shell eats commas.
REM
REM Skip the test suite on a docs-only change:  publish.bat "Fix typo" --fast

setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Could not find %PY%
    echo Create the virtual environment first, then run this again.
    pause
    exit /b 1
)

if "%~1"=="" (set "MSG=Update %DATE% %TIME%") else (set "MSG=%~1")

set "SKIP="
if /I "%~2"=="--fast" set "SKIP=--skip-tests"

echo.
echo === Preflight ===========================================================
"%PY%" scripts\preflight.py %SKIP%
if errorlevel 1 (
    echo.
    echo NOTHING WAS PUSHED. The live site is unchanged.
    echo Fix the problems listed above, then run publish again.
    pause
    exit /b 1
)

echo.
echo === Publishing ==========================================================
"%PY%" scripts\stamp_build.py

git add -A
git commit -m "%MSG%"
if errorlevel 1 (
    echo.
    echo Nothing to commit — the live site already has this code.
    pause
    exit /b 0
)

git push
if errorlevel 1 (
    echo.
    echo PUSH FAILED. The commit is saved locally but the site is unchanged.
    echo Check your GitHub sign-in, then run: git push
    pause
    exit /b 1
)

echo.
echo Pushed. Streamlit Cloud rebuilds in about a minute.
echo.
echo To confirm it actually landed:
echo   1. Open https://justychng-alpha-engine-dashboard.streamlit.app/
echo   2. Hard-refresh with Ctrl+Shift+R
echo   3. Check the build stamp at the bottom of the sidebar matches the one above
echo.
pause
