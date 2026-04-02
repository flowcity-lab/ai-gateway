@echo off
chcp 65001 >nul
echo ============================================
echo   AI-Gateway Deploy
echo   Repo: flowcity-lab/ai-gateway
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Dateien stagen...
git add .
if errorlevel 1 (
    echo FEHLER beim Stagen!
    goto :end
)
echo      OK
echo.

echo [2/3] Commit erstellen...
set /p MSG="Commit-Nachricht (Enter = 'update'): "
if "%MSG%"=="" set MSG=update
git commit -m "%MSG%"
if errorlevel 1 (
    echo Keine Aenderungen oder Fehler beim Commit.
    goto :end
)
echo      OK
echo.

echo [3/3] Push zu GitHub (Coolify baut automatisch)...
git push origin main
if errorlevel 1 (
    echo FEHLER beim Push! Bist du authentifiziert?
    goto :end
)
echo      OK
echo.

echo ============================================
echo   ERFOLGREICH! Coolify startet Rebuild.
echo   Pruefe Status auf Coolify Dashboard.
echo ============================================

:end
echo.
pause

