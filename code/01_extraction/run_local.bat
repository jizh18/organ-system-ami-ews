@echo off
cd /d %~dp0
start "rep2" cmd /k D:\Software\Conda\python.exe full_eval.py cv 2
start "heldout" cmd /k D:\Software\Conda\python.exe full_eval.py heldout
