@echo off
REM One-time setup: create venv + install the NFP package with GUI and dev extras
cd /d %~dp0..
py -3.12 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[gui,dev]"
echo.
echo Setup complete. Next:  scripts\run_demo.bat
pause
