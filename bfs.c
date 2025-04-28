#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node** createGraph(int n) {
    Node** adjList = (Node**)malloc(n * sizeof(Node*));
    for (int i = 0; i < n; i++) adjList[i] = NULL;
    return adjList;
}

void addEdge(Node** adjList, int u, int v) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = adjList[u];
    adjList[u] = newNode;

    newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = u;
    newNode->next = adjList[v];
    adjList[v] = newNode;
}

void generateRandomGraph(Node** adjList, int n) {
    srand(time(NULL));

    // Build a tree (guaranteed connected)
    for (int i = 1; i < n; i++) {
        int v = rand() % i;
        addEdge(adjList, i, v);
    }

    // Add more edges
    int extraEdges = n; // can be increased for denser graph
    for (int i = 0; i < extraEdges; i++) {
        int u = rand() % n;
        int v = rand() % n;
        if (u != v) addEdge(adjList, u, v);
    }
}

void bfsSequential(Node** adjList, int n, int start) {
    bool* visited = (bool*)calloc(n, sizeof(bool));
    int* queue = (int*)malloc(n * sizeof(int));
    int front = 0, rear = 0;

    visited[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int u = queue[front++];
        for (Node* temp = adjList[u]; temp; temp = temp->next) {
            int v = temp->vertex;
            if (!visited[v]) {
                visited[v] = true;
                queue[rear++] = v;
            }
        }
    }

    free(visited);
    free(queue);
}

void bfsParallel(Node** adjList, int n, int start) {
    bool* visited = (bool*)calloc(n, sizeof(bool));
    int* frontier = (int*)malloc(n * sizeof(int));
    int frontierSize = 0;

    visited[start] = true;
    frontier[frontierSize++] = start;

    while (frontierSize > 0) {
        int* nextFrontier = (int*)malloc(n * sizeof(int));
        int nextSize = 0;

        #pragma omp parallel
        {
            int* localQueue = (int*)malloc(n * sizeof(int));
            int localSize = 0;

            #pragma omp for nowait
            for (int i = 0; i < frontierSize; i++) {
                int u = frontier[i];
                for (Node* temp = adjList[u]; temp; temp = temp->next) {
                    int v = temp->vertex;
                    if (!visited[v]) {
                        #pragma omp critical
                        {
                            if (!visited[v]) {
                                visited[v] = true;
                                localQueue[localSize++] = v;
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < localSize; i++)
                    nextFrontier[nextSize++] = localQueue[i];
            }

            free(localQueue);
        }

        frontierSize = nextSize;
        for (int i = 0; i < nextSize; i++)
            frontier[i] = nextFrontier[i];

        free(nextFrontier);
    }

    free(visited);
    free(frontier);
}

int main() {
    int n;
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    Node** adjList = createGraph(n);
    generateRandomGraph(adjList, n);

    double start, end;

    start = omp_get_wtime();
    bfsSequential(adjList, n, 0);
    end = omp_get_wtime();
    double time_seq = end - start;
    printf("Sequential BFS Time: %.6f s\n", time_seq);

    start = omp_get_wtime();
    bfsParallel(adjList, n, 0);
    end = omp_get_wtime();
    double time_par = end - start;
    printf("Parallel BFS Time: %.6f s\n", time_par);
    printf("BFS Speedup: %.2fx\n", time_seq / time_par);

    // Free memory
    for (int i = 0; i < n; i++) {
        Node* curr = adjList[i];
        while (curr) {
            Node* next = curr->next;
            free(curr);
            curr = next;
        }
    }
    free(adjList);

    return 0;
}

