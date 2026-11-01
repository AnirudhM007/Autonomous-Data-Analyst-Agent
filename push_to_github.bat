@echo off
echo ========================================================
echo Pushing AUTONOMOUS DATA ANALYST AGENT to GitHub...
echo Repository: https://github.com/AnirudhM007/Autonomous-Data-Analyst-Agent
echo ========================================================
echo.
git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo [SUCCESS] Your full project has been uploaded to GitHub!
    echo Refresh your browser: https://github.com/AnirudhM007/Autonomous-Data-Analyst-Agent
    echo ========================================================
) else (
    echo.
    echo [ERROR] Push failed. If you have not created the repository yet, please create:
    echo        https://github.com/new with name "Autonomous-Data-Analyst-Agent"
    echo        and run this script again.
)
echo.
pause
