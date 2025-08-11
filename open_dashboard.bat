@echo off
echo Opening dashboard in browser...
timeout /t 2 /nobreak > nul
start http://localhost:5000
echo Dashboard opened in browser!
echo If page doesn't load, make sure the server is running (run start_dashboard.bat first)