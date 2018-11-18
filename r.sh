nvcc tests.cpp \
    ./sorting/sorting.test.cu   \
    -odir ./bin --expt-extended-lambda --generate-code arch=compute_52,code=sm_52 --run
