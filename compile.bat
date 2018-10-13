setEnvironment.bat && nvcc main.cu ./Performance/Performance.cpp ./genetics/Randomize.cu -O1 -Xptxas -O1 -Xcompiler -MP --run

REM -O3 -Xptxas -O3