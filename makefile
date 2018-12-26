
flags = --expt-extended-lambda

all: test

test: bin/tests.o bin/sorting.test.o bin/crowdingDistance.test.o bin/selection.test.o bin/crossover.test.o
	nvcc bin/tests.o bin/sorting.test.o bin/crowdingDistance.test.o bin/selection.test.o bin/selection.o bin/crossover.test.o -o bin/a.out

bin/tests.o: tests.cpp
	g++ -c tests.cpp -o bin/tests.o

bin/sorting.test.o: ./sorting/sorting.test.cu ./sorting/sorting.cuh
	nvcc $(flags) -c ./sorting/sorting.test.cu -o bin/sorting.test.o

bin/crowdingDistance.test.o: ./crowdingDistance/crowdingDistance.test.cu ./crowdingDistance/crowdingDistance.cuh
	nvcc $(flags) -c ./crowdingDistance/crowdingDistance.test.cu -o bin/crowdingDistance.test.o

bin/selection.test.o: ./nsga/selection.test.cu ./nsga/selection.cuh bin/selection.o
	nvcc $(flags) -c ./nsga/selection.test.cu -o bin/selection.test.o


bin/selection.o: ./nsga/selection.cu
	nvcc $(flags) -c ./nsga/selection.cu -o bin/selection.o

bin/crossover.test.o: ./nsga/crossover.test.cu ./nsga/crossover.cuh
	nvcc $(flags) -c ./nsga/crossover.test.cu -o bin/crossover.test.o

clean:
	rm ./bin/*