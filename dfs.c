#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>

#define MAX_THREADS 8

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct {
    Node** adjList;
    int V;
} Graph;

Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->adjList = (Node**)malloc(V * sizeof(Node*));
    for (int i = 0; i < V; i++) graph->adjList[i] = NULL;
    return graph;
}

void addEdge(Graph* graph, int src, int dest) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = dest;
    newNode->next = graph->adjList[src];
    graph->adjList[src] = newNode;

    newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = src;
    newNode->next = graph->adjList[dest];
    graph->adjList[dest] = newNode;
}

void generateRandomTree(Graph* graph, int V) {
    for (int i = 1; i < V; i++) {
        int parent = rand() % i;
        addEdge(graph, parent, i);
    }
}

void DFSSequential(Graph* graph, int start, bool* visited) {
    int* stack = (int*)malloc(graph->V * sizeof(int));
    int top = -1;
    stack[++top] = start;

    while (top >= 0) {
        int curr = stack[top--];
        if (!visited[curr]) {
            visited[curr] = true;
            Node* temp = graph->adjList[curr];
            while (temp) {
                if (!visited[temp->vertex])
                    stack[++top] = temp->vertex;
                temp = temp->next;
            }
        }
    }
    free(stack);
}

void DFSParallel(Graph* graph, int start, bool* visited) {
    int* frontier = (int*)malloc(graph->V * sizeof(int));
    int frontierSize = 1;
    frontier[0] = start;
    omp_lock_t* locks = (omp_lock_t*)malloc(graph->V * sizeof(omp_lock_t));
    for (int i = 0; i < graph->V; i++) omp_init_lock(&locks[i]);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_THREADS)
    for (int i = 0; i < frontierSize; i++) {
        int* localStack = (int*)malloc(graph->V * sizeof(int));
        int top = -1;
        localStack[++top] = frontier[i];

        while (top >= 0) {
            int curr = localStack[top--];

            if (!visited[curr]) {
                omp_set_lock(&locks[curr]);
                if (!visited[curr]) {
                    visited[curr] = true;
                    Node* temp = graph->adjList[curr];
                    while (temp) {
                        if (!visited[temp->vertex])
                            localStack[++top] = temp->vertex;
                        temp = temp->next;
                    }
                }
                omp_unset_lock(&locks[curr]);
            }
        }
        free(localStack);
    }
    for (int i = 0; i < graph->V; i++) omp_destroy_lock(&locks[i]);
    free(locks);
    free(frontier);
}

int main() {
    int V;
    printf("Enter number of vertices: ");
    scanf("%d", &V);

    Graph* graph = createGraph(V);
    generateRandomTree(graph, V);

    bool* visitedSeq = (bool*)calloc(V, sizeof(bool));
    bool* visitedPar = (bool*)calloc(V, sizeof(bool));

    double startTime, endTime;

    startTime = omp_get_wtime();
    DFSSequential(graph, 0, visitedSeq);
    endTime = omp_get_wtime();
    double seqTime = endTime - startTime;
    printf("Sequential DFS Time: %f seconds\n", seqTime);

    startTime = omp_get_wtime();
    DFSParallel(graph, 0, visitedPar);
    endTime = omp_get_wtime();
    double parTime = endTime - startTime;
    printf("Parallel DFS Time:   %f seconds\n", parTime);

    if (parTime > 0)
        printf("Speedup: %.2fx\n", seqTime / parTime);

    free(visitedSeq);
    free(visitedPar);
    for (int i = 0; i < V; i++) {
        Node* temp = graph->adjList[i];
        while (temp) {
            Node* toFree = temp;
            temp = temp->next;
            free(toFree);
        }
    }
    free(graph->adjList);
    free(graph);
    return 0;
}
