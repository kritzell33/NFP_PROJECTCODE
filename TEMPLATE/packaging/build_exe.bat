@echo off
REM Build the NFP Assessor .exe (run on Windows from anywhere; cds to repo root)
cd /d %~dp0..
call .venv\Scripts\activate.bat
pyinstaller packaging\assessor.spec --noconfirm
echo.
echo Build finished. Launch: dist\NFP_Assessor\NFP_Assessor.exe
pause
