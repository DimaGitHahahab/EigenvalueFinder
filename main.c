#include "return_codes.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define epsilon 1e-8

int readMatrix(const char *filename, double **A, int *n);

int writeEigenvalues(const char *filename, double *A, int n);

double dotProduct(const double *a, const double *b, int n);

void normalizeVector(double *v, int n);

void projection(const double *v, const double *u, double *result, int n);

void transposeMatrix(const double *Q, double *QT, int n);

int qrDecomposition(const double *A, double *Q, double *R, int n);

void multiplyMatrices(const double *A, const double *B, double *C, int n);

int isConverged(const double *A, int n);

// Reads matrix from input file, calculates eigenvalues and writes them to output file
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Wrong number of arguments\n");
        return ERROR_PARAMETER_INVALID;
    }

    double *A = NULL;
    int n;
    int result;

    if ((result = readMatrix(argv[1], &A, &n)) != SUCCESS) {
        fprintf(stderr, "Cannot read matrix from file");
        return result;
    }

    int max_iterations = 1500;

    double *Q = (double *) malloc(sizeof(double) * n * n);
    if (Q == NULL) {
        free(A);
        fprintf(stderr, "Cannot allocate memory");
        return ERROR_OUT_OF_MEMORY;
    }
    double *R = (double *) malloc(sizeof(double) * n * n);
    if (R == NULL) {
        free(A);
        free(Q);
        fprintf(stderr, "Cannot allocate memory");
        return ERROR_OUT_OF_MEMORY;
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        result = qrDecomposition(A, Q, R, n);
        if (result != SUCCESS) {
            free(A);
            free(Q);
            free(R);
            fprintf(stderr, "QR decomposition failed");
            return result;
        }

        multiplyMatrices(R, Q, A, n);

        if (isConverged(A, n)) {
            break;
        }
    }

    free(Q);
    free(R);

    if ((result = writeEigenvalues(argv[2], A, n)) != SUCCESS) {
        fprintf(stderr, "Cannot write eigenvalues to file");
        return result;
    }

    free(A);

    return result;
}

// Reads matrix from file and stores it in A
int readMatrix(const char *filename, double **A, int *n) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        free(*A);

        return ERROR_CANNOT_OPEN_FILE;
    }

    if (fscanf(file, "%d", n) != 1) {
        free(*A);
        fclose(file);
        return ERROR_DATA_INVALID;
    }

    if (*n <= 0) {
        free(*A);
        fclose(file);
        return ERROR_PARAMETER_INVALID;
    }

    *A = (double *) malloc(sizeof(**A) * (*n) * (*n));
    if (*A == NULL) {
        free(*A);
        fclose(file);
        return ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < (*n) * (*n); i++) {
        if (fscanf(file, "%lf", (*A) + i) != 1) {
            free(*A);
            fclose(file);
            return ERROR_DATA_INVALID;
        }
    }

    fclose(file);
    return SUCCESS;
}

// Calculates scalar product of two vectors
double dotProduct(const double *a, const double *b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Calculates projection of vector v onto vector u
void projection(const double *v, const double *u, double *result, int n) {
    double dot_vu = dotProduct(v, u, n);
    double dot_uu = dotProduct(u, u, n);
    if (dot_uu == 0.0) {
        for (int i = 0; i < n; i++) {
            result[i] = 0.0;
        }
        return;
    }
    double scale = dot_vu / dot_uu;
    for (int i = 0; i < n; i++) {
        result[i] = scale * u[i];
    }
}

// Normalizes vector v
void normalizeVector(double *v, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    if (fabs(norm) < epsilon) {
        for (int i = 0; i < n; i++) {
            v[i] = 0.0;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        v[i] /= norm;
    }
}

// Transposes Q and stores it in QT
void transposeMatrix(const double *Q, double *QT, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            QT[i * n + j] = Q[j * n + i];
        }
    }
}

// Multiply matrices A and B, and store the result in C
void multiplyMatrices(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Calculates Q
int GramSchmidtProcess(const double *A, double *Q, int n) {
    double *temp = (double *) malloc(sizeof(double) * n);
    if (temp == NULL) {
        return ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[i * n + j] = A[i * n + j];
        }
        for (int k = 0; k < i; k++) {
            projection(&A[i * n], &Q[k * n], temp, n);
            for (int j = 0; j < n; j++) {
                Q[i * n + j] -= temp[j];
            }
        }
        normalizeVector(&Q[i * n], n);
    }
    free(temp);

    double *tempTest = (double *) malloc(sizeof(double) * n * n);
    if (tempTest == NULL) {
        return ERROR_OUT_OF_MEMORY;
    }
    transposeMatrix(Q, tempTest, n);
    memcpy(Q, tempTest, sizeof(double) * n * n);
    free(tempTest);
    return SUCCESS;
}

// Checks if the matrix converged enough
int isConverged(const double *arr, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && fabs(arr[i * n + j]) > epsilon) {
                return 0;
            }
        }
    }
    return 1;
}

// Calculates new A matrix
int qrDecomposition(const double *A, double *Q, double *R, int n) {
    int result = GramSchmidtProcess(A, Q, n);
    if (result != SUCCESS) {
        return result;
    }

    double *QT = (double *) malloc(sizeof(double) * n * n);
    if (QT == NULL) {
        return ERROR_OUT_OF_MEMORY;
    }
    transposeMatrix(Q, QT, n);

    multiplyMatrices(QT, A, R, n);
    free(QT);
    return SUCCESS;
}

// Calculates determinant of 2x2 matrix
double findDeterminant(double a11, double a12, double a21, double a22) {
    return a11 * a22 - a12 * a21;
}

// Writes eigenvalues to output file
int writeEigenvalues(const char *filename, double *A, int n) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        return ERROR_CANNOT_OPEN_FILE;
    }
    double real;
    double imag;
    double determinant;
    for (int i = 0; i < n; i++) {
        if (fabs(A[(i + 1) * n + i]) > epsilon) {
            real = (A[i * n + i] + A[(i + 1) * n + i + 1]) / 2;
            determinant = findDeterminant(A[i * n + i], A[i * n + i + 1], A[(i + 1) * n + i], A[(i + 1) * n + i + 1]);
            imag = sqrt(fabs(real * real - determinant));
            fprintf(file, "%g +%gi\n", real, imag);
            fprintf(file, "%g -%gi\n", real, imag);
            i++;
        } else {
            fprintf(file, "%g\n", A[i * n + i]);
        }
    }

    fclose(file);
    return SUCCESS;
}