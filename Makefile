SRC := dijkstra.c
MPISRC := dijkstra_mpi.c
OMPSRC := dijkstra_omp.c

default: dijkstra

dijkstra: $(SRC)
	gcc -O3 -Wall -Wextra -o $@ $<

dijkstra_mpi: $(MPISRC)
	mpicc -O3 -Wall -Wextra -o $@ $<

dijkstra_omp: $(OMPSRC)
	gcc -fopenmp -O3 -Wall -Wextra -o $@ $<

clean: 
	rm -f dijkstra dijkstra_mpi dijkstra_omp
