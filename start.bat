@echo off
cd /d "%~dp0"

echo ============================================
echo  ECML Inpaint Studio
echo ============================================
echo.
echo Starting services (this may take 2-3 minutes
echo while models load)...
echo.

docker compose up -d --wait

echo.
echo Services ready! Opening browser...
start http://localhost:3000
start http://localhost:8188
echo.
echo  App:     http://localhost:3000
echo  ComfyUI: http://localhost:8188
echo.
echo Press any key to stop all services...
pause > nul
docker compose down
