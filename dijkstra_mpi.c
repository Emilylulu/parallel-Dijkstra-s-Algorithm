/* assert */
#include <assert.h>
/* INFINITY */
#include <math.h>
/* FILE, fopen, fclose, fscanf, rewind */
#include <stdio.h>
/* EXIT_SUCCESS, malloc, calloc, free */
#include <stdlib.h>
/* time, CLOCKS_PER_SEC */
#include <time.h>
#include <mpi.h>
#include <memory.h>

#define ROWMJR(R, C, NR, NC) (R*NC+C)
#define COLMJR(R, C, NR, NC) (C*NR+R)
/* define access directions for matrices */
#define a(R, C) a[ROWMJR(R,C,ln,n)]
#define b(R, C) b[ROWMJR(R,C,nn,n)]
#define MAIN_PROCESS 0
#define SEND_NUM_TAG 0
#define SEND_DISPLS_TAG 1
#define SEND_ELEMNTS_TAG 2
#define SEND_WEIGHT_TAG 3
#define SEND_COUNTS_TAG 4
#define SEND_RESULT_TAG 5

static void calculateDispls(int ** displs, int ** localNumOfElements, int numberOfProcessors, int size){
    int local, reminder;
    if (size < numberOfProcessors){//If there are more nodes than rows(columns), the extra node(s) are useless.
        local = 1;
        *localNumOfElements = malloc(numberOfProcessors * sizeof(**localNumOfElements));
        int i = 0;
        for (; i < size; i++){
            *(*localNumOfElements + i) = local;
        }
        for (; i < numberOfProcessors;i++){
            *(*localNumOfElements + i) = 0;
        }
    } else {//If there are more rows than available nodes, separate into almost equal pieces.
        local = size / numberOfProcessors;
        reminder = size % numberOfProcessors;

        //calculate numbers to send to each node.
        *localNumOfElements = malloc(numberOfProcessors * sizeof(**localNumOfElements));
        for (int i = 0; i < numberOfProcessors; i++) {
            *(*localNumOfElements + i) = local;
        }
        *(*localNumOfElements + numberOfProcessors - 1) += reminder;
    }
    //calculate offset that points from send buffer(vals).
    *displs = malloc(numberOfProcessors * sizeof(**displs));
    for (int i = 0; i < numberOfProcessors; i++) {
        *(*displs + i) = 0;
        for (int j = 0; j < i; j++) {
            *(*displs + i) += *(*localNumOfElements + j);
        }
    }
}
static void
load(
        char const *const filename,
        int *const np,
        float **const ap, int numberOfProcessors, int ** displs, int ** localNumOfElements, int rank
) {
    int n;
    float *a = NULL;
    if (rank == MAIN_PROCESS) {//Main node read the file and send to other nodes piece by piece.
        int i, j, k, ret;
        FILE *fp = NULL;


        /* open the file */
        fp = fopen(filename, "r");
        assert(fp);

        /* get the number of nodes in the graph */
        ret = fscanf(fp, "%d", &n);
        assert(1 == ret);

        //Calculate how many rows each node will hold. And their offsets(displs).
        calculateDispls(displs, localNumOfElements, numberOfProcessors, n);

        /* allocate memory for local values */
        a = malloc(n * *(*localNumOfElements) * sizeof(*a));
        for (j = 0; j < *(* localNumOfElements) * n; ++j) {
            ret = fscanf(fp, "%f", &a[j]);
            assert(1 == ret);
        }
        *ap = a;
        //printf("0. %d info read\n", j);//TODO debug use only
        //Read & send. Each node will get localNumOfElements rows of data.
        for (i = 1; i < numberOfProcessors; ++i) {
            a = malloc(n * *(*localNumOfElements + i) * sizeof(*a));
            //Send just collected info to that node
            MPI_Send(&n, 1, MPI_INTEGER, i, SEND_NUM_TAG, MPI_COMM_WORLD);

            MPI_Send(*displs, numberOfProcessors, MPI_INTEGER, i, SEND_DISPLS_TAG, MPI_COMM_WORLD);
            MPI_Send(*localNumOfElements, numberOfProcessors, MPI_INTEGER, i, SEND_ELEMNTS_TAG, MPI_COMM_WORLD);
            //Read file
            for (j = 0; j < *(* localNumOfElements + i) * n; ++j) {
                ret = fscanf(fp, "%f", &a[j]);
                assert(1 == ret);
            }
            //printf("%d. %d info read\n", i, j);//TODO debug use only
            MPI_Send(&j, 1, MPI_INTEGER, i, SEND_COUNTS_TAG, MPI_COMM_WORLD);
            MPI_Send(a, j, MPI_FLOAT, i, SEND_WEIGHT_TAG, MPI_COMM_WORLD);
            free(a);//Free memory after send.
        }

        /* close file */
        ret = fclose(fp);
        assert(!ret);
    } else {//All nodes except MAIN NODE will receive piece(rows) of graph data.
        int count;
        MPI_Recv(&n, 1, MPI_INTEGER, MAIN_PROCESS, SEND_NUM_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *displs = malloc(numberOfProcessors * sizeof(**displs));
        MPI_Recv(*displs, numberOfProcessors, MPI_INTEGER, MAIN_PROCESS, SEND_DISPLS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *localNumOfElements = malloc(numberOfProcessors * sizeof(**localNumOfElements));
        MPI_Recv(*localNumOfElements, numberOfProcessors, MPI_INTEGER, MAIN_PROCESS, SEND_ELEMNTS_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(&count, 1, MPI_INTEGER, MAIN_PROCESS, SEND_COUNTS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        a = malloc(count * sizeof(*a));
        MPI_Recv(a, count, MPI_FLOAT, MAIN_PROCESS, SEND_WEIGHT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *ap = a;
    }
    /* record output values */

    *np = n;
    MPI_Barrier(MPI_COMM_WORLD);
}

static void
dijkstra(
        int const source,
        int const n,
        float const *const a,
        float **const result, int rank, int * displs, int * localNumOfElements, int numberOfProcessors
) {
    int i, j, k, sourceNode = 0;
    struct float_int {
        float distance;
        int u;
    } min;
    char *set = NULL;
    float *resultVector = NULL;
    float * localResult = NULL;

    //Simple hash set to record which vertex has already been visited(with fixed min distance).
    //0 stands for not visited. Other values stands for visited.
    set = calloc(n, sizeof(*set));
    assert(set);

    //A vector(1d array) to store for result(the min distance from source to each vertex).
    resultVector = malloc(n * sizeof(*resultVector));
    assert(resultVector);

    localResult = malloc(n * sizeof(*resultVector));
    assert(localResult);

    for (i = 0; i < numberOfProcessors; i++){
        if (source < displs[i]){
            sourceNode = i - 1;
            break;
        }
    }
    
    //The initial result distance is what currently the distance between source to each vertex.
    if (rank == sourceNode) {//Copy and prepare for broad cast.
        for (i = 0; i < n; ++i) {
            resultVector[i] = a[i + n * (source - displs[sourceNode])];
            
        }
    }
    MPI_Bcast(resultVector, n, MPI_FLOAT, sourceNode, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //We don't have to visit source vertex.
    //Such that the distance from source to source is defined in the graph_matrix[source, source].
    set[source] = 1;
    min.u = -1; /* avoid compiler warning */

    //Iterate through each vertex.
    for (i = 1; i < n; ++i) {
        min.distance = INFINITY;
        //Find local minimum vertex in order to do relax operation(update min distance to each vertex that vertex connected with).
        for (j = 0; j < n; ++j) {
            if (!set[j] && resultVector[j] < min.distance) {
                min.distance = resultVector[j];
                min.u = j;
            }
            localResult[j] = resultVector[j];
        }
       
        set[min.u] = 1;
        for (j = 0; j < localNumOfElements[rank]; j++){
            if (set[j + displs[rank]]){
                continue;
            }
            if (a(j, min.u) + min.distance < localResult[j + displs[rank]]){
                localResult[j + displs[rank]] = a(j, min.u) + min.distance;
            }
            //printf("Vertex %d: Local min is: %.1f\n", j, localResult[j + displs[rank]]);
        }

        MPI_Allreduce(localResult, resultVector, n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(set);

    *result = resultVector;
}

static void
print_time(double const seconds) {
    printf("Operation Time: %0.04fs\n", seconds);
}

static void
print_numbers(
        char const *const filename,
        int const n,
        float const *const numbers) {
    int i;
    FILE *fout;

    /* open file */
    if (NULL == (fout = fopen(filename, "w"))) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    /* write numbers to fout */
    for (i = 0; i < n; ++i) {
        fprintf(fout, "%10.4f\n", numbers[i]);
    }

    fclose(fout);
}

int
main(int argc, char **argv) {
    int n, numberOfProcessors, rank;;
    double ts, te;
    float *a = NULL, *result = NULL;
    int * displs = NULL, *localNumOfElements = NULL;

    if (argc < 4) {
        printf("Invalid number of arguments.\nUsage: dijkstra <graph> <source> <output_file>.\n");
        return EXIT_FAILURE;
    }


    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    load(argv[1], &n, &a, numberOfProcessors, &displs, &localNumOfElements, rank);
    /* Debug load function use
    char fileName[50];
    sprintf(fileName, "output%d", rank);
    print_numbers(fileName, *(localNumOfElements + rank) * n, a);
     */
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    dijkstra(atoi(argv[2]), n, a, &result, rank, displs, localNumOfElements, numberOfProcessors);
    te = MPI_Wtime();

    //print_time((te - ts) / CLOCKS_PER_SEC);
    if (rank == MAIN_PROCESS) {
        print_time(te - ts);
        print_numbers(argv[3], n, result);
    }
    free(a);
    free(result);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
