/*
    How to compile and run this code:
    Compile: mpicc -o matrix_mpi matrix_mul_mpi.c
    Run:     mpirun -np 2 ./matrix_mpi

    This program performs matrix multiplication using MPI, distributing the work across multiple processes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to print a matrix (for debugging purposes)
void display(int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // Initialize the MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    // Matrix dimensions
    int K = 100, M = 50, N = 50, P = 50;

    // Broadcasting the matrix dimensions to all processes
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Ensure the number of matrices is divisible by the number of processes
    if(K % size != 0) {
        printf("Number of matrices must be divisible by the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Declare matrices: A (K x M x N), B (K x N x P), and result R (K x M x P)
    int A[K][M][N], B[K][N][P], R[K][M][P];

    // Initialize matrices A and B in the root process (rank 0)
    if(rank == 0) {
        // Initialize matrix A with random values between 0 and 99
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                    A[k][i][j] = rand() % 100;
                }
            }
        }
        // Initialize matrix B with random values between 0 and 99
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < P; j++) {
                    B[k][i][j] = rand() % 100;
                }
            }
        }
    }

    // Buffers to store portions of the matrices that each process will work on
    int localA[K / size][M][N], localB[K / size][N][P], localR[K / size][M][P];
    
    // Scatter matrices A and B to all processes
    MPI_Scatter(A, (K / size) * M * N, MPI_INT, localA, (K / size) * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, (K / size) * N * P, MPI_INT, localB, (K / size) * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Start the timer for performance measurement
    double startTime = MPI_Wtime();

    // Perform matrix multiplication (local computation for each process)
    for(int k = 0; k < (K / size); k++) {  // Each process handles a portion of K matrices
        for(int i = 0; i < M; i++) {         // Iterate over rows of matrix A
            for(int j = 0; j < P; j++) {     // Iterate over columns of matrix B
                localR[k][i][j] = 0;  // Initialize the result element to 0
                for(int l = 0; l < N; l++) {  // Perform dot product for multiplication
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) % 100;  // Modulo 100
                }
                localR[k][i][j] %= 100;  // Take modulo 100 for the final result
            }
        }
    }

    // End the timer for performance measurement
    double endTime = MPI_Wtime();

    // Gather the result matrices from all processes to the root process
    MPI_Gather(localR, (K / size) * M * P, MPI_INT, R, (K / size) * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Remove the comment to print result matrices for debugging (in root process)
    // if(rank == 0) {
    //     for(int k = 0; k < K; k++) {
    //         printf("Result Matrix R%d\n", k);
    //         display(M, P, R[k]);  // Print the result of matrix multiplication
    //     }
    // }

    // Barrier to synchronize all processes before printing timing information
    MPI_Barrier(MPI_COMM_WORLD);

    // Print time taken by each process (useful for performance analysis)
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    MPI_Finalize();  // Finalize the MPI environment
    return 0;
}
