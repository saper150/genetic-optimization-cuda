setEnvironment.bat && nvcc main.cu ./Performance/Performance.cpp ./genetics/Randomize.cu ./fitness/test.cu --expt-extended-lambda -Xcompiler -MP --run

REM -O3 -Xptxas -O3