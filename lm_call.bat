@echo off
cls
set /a Counter=0
:start
echo off
C:\Users\cel_w\Anaconda2\python.exe C:\Users\cel_w\Documents\GitHub\ir2qsuggestion\lambda_mart.py
set /a Counter+=1
echo %Counter%
goto start