setEnvironment.bat && nvcc tests.cpp^
    ./sorting/sorting.test.cu^
    -odir ./bin --expt-extended-lambda --generate-code arch=compute_52,code=sm_52 -Xcompiler -MP --run
    REM ./fitness/KnnFitness.test.cu^
    REM ./fitness/fitnessTransform.cu^
    REM ./Performance/Performance.cpp^
    REM ./genetics/Randomize.cu^
    REM ./Knn/Knn.test.cu^
    REM ./fitness/populationReduction.test.cu^
    REM ./genetics/genetics.test.cu^

REM -O3 -Xptxas -O3