@echo off
cd /d "%~dp0"
echo Starting Stroke Risk Intelligence dashboard...
echo.
echo Keep this window open while using the dashboard.
echo Browser URL: http://127.0.0.1:8501
echo.
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
pause
