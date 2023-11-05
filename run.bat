@echo off

echo don't forget to run install bat and make sure that you have changed the paths in resizer_v3.py file
echo moreover set your thread count according to your CPU
call .\venv\Scripts\activate.bat

set CUDA_VISIBLE_DEVICES=0
python resizer_v4.py

pause
