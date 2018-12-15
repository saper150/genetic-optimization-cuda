
flags = --expt-extended-lambda

all: test

test: bin/tests.o bin/sorting.test.o bin/crowdingDistance.test.o
	nvcc bin/tests.o bin/sorting.test.o bin/crowdingDistance.test.o -o bin/a.out

bin/tests.o: tests.cpp
	g++ -c tests.cpp -o bin/tests.o

bin/sorting.test.o: ./sorting/sorting.test.cu ./sorting/sorting.cuh
	nvcc $(flags) -c ./sorting/sorting.test.cu -o bin/sorting.test.o

bin/crowdingDistance.test.o: ./crowdingDistance/crowdingDistance.test.cu ./crowdingDistance/crowdingDistance.cuh
	nvcc $(flags) -c ./crowdingDistance/crowdingDistance.test.cu -o bin/crowdingDistance.test.o

clean:
	rm ./bin/*