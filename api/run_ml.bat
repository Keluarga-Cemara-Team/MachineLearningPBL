@echo off
title AntiBurnout - Machine Learning API Service (Port 8000)
cd /d "%~dp0"

echo ======================================================================
echo    Memulai AntiBurnout Machine Learning Microservice API
echo    Model Champion: Gaussian Naive Bayes (Akurasi: 98.90%%)
echo ======================================================================

set PYTHON_CMD=python
if exist "C:\laragon\bin\python\python-3.10\python.exe" (
    set "PYTHON_CMD=C:\laragon\bin\python\python-3.10\python.exe"
)

"%PYTHON_CMD%" run.py
pause
