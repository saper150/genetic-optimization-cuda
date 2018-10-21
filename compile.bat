setEnvironment.bat && nvcc tests.cpp ./Performance/Performance.cpp ./genetics/Randomize.cu ./Knn/Knn.test.cu ./fitness/populationReduction.test.cu ./genetics/genetics.test.cu -odir ./bin --expt-extended-lambda --generate-code arch=compute_52,code=sm_52 -Xcompiler -MP --run

REM -O3 -Xptxas -O3