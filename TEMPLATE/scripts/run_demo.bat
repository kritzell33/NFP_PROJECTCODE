@echo off
REM Run the bundled sample scenario (100 replications) and open the report
cd /d %~dp0..
call .venv\Scripts\activate.bat
nfp run --runs 100 --seed 42 --open
pause
