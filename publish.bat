@echo off
REM Commit every local change and push it live. Streamlit Cloud redeploys
REM automatically within about a minute of the push landing.
cd /d "%~dp0"

set MSG=%*
if "%MSG%"=="" set MSG=Update %DATE% %TIME%

git add -A
git commit -m "%MSG%"
git push

echo.
echo Pushed. The live site rebuilds in ~1 minute.
pause
