// merge_sort_simplified.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>    // Include OpenMP header
#include <time.h>   // For timing and seeding random numbers

// --- The Merge Function ---
// Merges two **already sorted** subarrays back into the main array.
// Subarray 1: arr[startIndex] ... arr[midIndex]
// Subarray 2: arr[midIndex + 1] ... arr[endIndex]
void merge(int *arrayToSort, int startIndex, int midIndex, int endIndex) {
    // Calculate sizes of the two temporary subarrays
    int sizeLeft = midIndex - startIndex + 1;
    int sizeRight = endIndex - midIndex;

    // Create temporary arrays to hold the left and right halves
    int *leftHalf = (int *)malloc(sizeLeft * sizeof(int));
    int *rightHalf = (int *)malloc(sizeRight * sizeof(int));

    // --- Copy data to temporary arrays ---
    // Copy elements from the main array into the left temporary array
    for (int i = 0; i < sizeLeft; i++) {
        leftHalf[i] = arrayToSort[startIndex + i];
    }
    // Copy elements from the main array into the right temporary array
    for (int j = 0; j < sizeRight; j++) {
        rightHalf[j] = arrayToSort[midIndex + 1 + j];
    }

    // --- Merge the temporary arrays back into the main array ---
    int indexLeft = 0;    // Initial index for the left temporary array
    int indexRight = 0;   // Initial index for the right temporary array
    int indexMerged = startIndex; // Initial index for the main array where merging starts

    // Compare elements from leftHalf and rightHalf and place the smaller one
    // into the main array. Continue until one of the temporary arrays is empty.
    while (indexLeft < sizeLeft && indexRight < sizeRight) {
        if (leftHalf[indexLeft] <= rightHalf[indexRight]) {
            arrayToSort[indexMerged] = leftHalf[indexLeft];
            indexLeft++;
        } else {
            arrayToSort[indexMerged] = rightHalf[indexRight];
            indexRight++;
        }
        indexMerged++;
    }

    // --- Copy any remaining elements ---
    // If there are elements left in leftHalf, copy them
    while (indexLeft < sizeLeft) {
        arrayToSort[indexMerged] = leftHalf[indexLeft];
        indexLeft++;
        indexMerged++;
    }
    // If there are elements left in rightHalf, copy them
    while (indexRight < sizeRight) {
        arrayToSort[indexMerged] = rightHalf[indexRight];
        indexRight++;
        indexMerged++;
    }

    // Free the memory allocated for the temporary arrays
    free(leftHalf);
    free(rightHalf);
}

// --- Sequential Merge Sort ---
// Standard recursive merge sort algorithm.
void sequentialMergeSort(int *arrayToSort, int startIndex, int endIndex) {
    // Base case: If the segment has 0 or 1 elements, it's already sorted.
    if (startIndex < endIndex) {
        // Find the middle point to divide the array
        int midIndex = startIndex + (endIndex - startIndex) / 2; // Avoids overflow for large indices

        // Recursively sort the first half
        sequentialMergeSort(arrayToSort, startIndex, midIndex);
        // Recursively sort the second half
        sequentialMergeSort(arrayToSort, midIndex + 1, endIndex);

        // Merge the two sorted halves
        merge(arrayToSort, startIndex, midIndex, endIndex);
    }
}

// --- Parallel Merge Sort using OpenMP Tasks ---

// Threshold: If the subarray size is below this, use sequential sort.
// This avoids the overhead of creating very small parallel tasks.
#define MIN_SIZE_FOR_PARALLEL 5000

void parallelMergeSort(int *arrayToSort, int startIndex, int endIndex) {
    // Check if the current segment is large enough for parallel execution
    if (startIndex < endIndex) {
        if ((endIndex - startIndex + 1) < MIN_SIZE_FOR_PARALLEL) {
            // --- Fallback Case ---
            // If the segment is too small, sort it sequentially.
            sequentialMergeSort(arrayToSort, startIndex, endIndex);
        } else {
            // --- Parallel Case ---
            // Find the middle point
            int midIndex = startIndex + (endIndex - startIndex) / 2;

            // *** Create Parallel Tasks ***
            // These tasks are like "job descriptions" added to a pool.
            // Available threads in the OpenMP team will pick up these tasks.

            #pragma omp task // Define the first task: sort the left half
            {
                // This block will be executed as an independent task
                parallelMergeSort(arrayToSort, startIndex, midIndex);
            } // Task for left half ends here

            #pragma omp task // Define the second task: sort the right half
            {
                // This block will also be executed as an independent task
                // Potentially at the same time as the left half task on another core
                parallelMergeSort(arrayToSort, midIndex + 1, endIndex);
            } // Task for right half ends here

            // *** Synchronization Point ***
            // #pragma omp taskwait: IMPORTANT!
            // The current thread will PAUSE here until BOTH of the child tasks
            // created *directly within this block* (sorting left and right halves)
            // have fully completed. We NEED both halves sorted before we can merge.
            #pragma omp taskwait

            // --- Merge Step ---
            // Only after both halves are confirmed sorted (due to taskwait),
            // merge them together. This merge is done by the current thread.
            merge(arrayToSort, startIndex, midIndex, endIndex);
        }
    }
}


// --- Utility Functions ---

// Fills an array with random integers.
void generateRandomArray(int *arr, int n) {
    for (int i = 1; i < n; i++) arr[i] = rand() % 10000000; // Numbers between 0 and 9999999
}

// Prints the first few elements of an array.
void printArray(int *arr, int n) {
    int limit = (n < 20) ? n : 20; // Print max 20 elements
    for (int i = 0; i < limit; i++)
        printf("%5d ", arr[i]);
    if (n > 20) printf("...");
    printf("\n");
}

// --- Main Function ---
int main() {
    int n; // Number of elements
    printf("Enter number of elements: ");
    scanf("%d", &n);

    // Allocate memory for two copies of the array
    int *arraySequential = (int *)malloc(n * sizeof(int));
    int *arrayParallel = (int *)malloc(n * sizeof(int));
    if (!arraySequential || !arrayParallel) {
        perror("Failed to allocate memory");
        return 1;
    }

    // Seed the random number generator
    srand(time(NULL));
    generateRandomArray(arraySequential, n);

    // Copy the unsorted data to the second array for parallel sorting
    for (int i = 0; i < n; i++) {
        arrayParallel[i] = arraySequential[i];
    }

    printf("\nOriginal array:\n");
    printArray(arraySequential, n);

    // --- Time and Run Sequential Sort ---
    printf("\nRunning Sequential Merge Sort...\n");
    double startTimeSeq = omp_get_wtime(); // Get start time
    sequentialMergeSort(arraySequential, 0, n - 1);
    double endTimeSeq = omp_get_wtime();   // Get end time
    double timeSequential = endTimeSeq - startTimeSeq;

    // --- Time and Run Parallel Sort ---
    printf("Running Parallel Merge Sort...\n");
    double startTimePar = omp_get_wtime(); // Get start time

    // *** Setting up the Parallel Region for Tasks ***
    // We need to create a team of threads *once* outside the recursive function.
    // The `#pragma omp parallel` directive creates the team.
    // The `#pragma omp single` directive ensures that ONLY ONE thread
    // from the team makes the *initial* call to parallelMergeSort.
    // That single call will then generate all the necessary tasks for the team to execute.
  
      // *** Optimization: Check if parallel region is even needed ***
    // If the total array size is smaller than the threshold where
    // parallelMergeSort would just call sequentialMergeSort anyway,
    // skip creating the parallel region to avoid its overhead.
    if (n > MIN_SIZE_FOR_PARALLEL) {
        // --- Use Parallel Execution ---
        #pragma omp parallel // Create the thread team
        {
            #pragma omp single // Only one thread starts the recursive task generation
            {
                parallelMergeSort(arrayParallel, 0, n - 1);
            } // End of single region
        } // End of parallel region (threads synchronize and disband here)
    } else {
        // --- Fallback for Small Arrays ---
        // The array is too small, just run the sequential version directly.
        // (Calling parallelMergeSort here would also work, as it would
        // immediately call sequentialMergeSort, but calling sequential directly
        // slightly more clearly shows what's happening).
        printf(" (Array size <= threshold, running sequentially)\n");
        sequentialMergeSort(arrayParallel, 0, n - 1);
        // Alternatively, you could still call parallelMergeSort here, it would just fall back:
        // parallelMergeSort(arrayParallel, 0, n - 1);
    }
    double endTimePar = omp_get_wtime();   // Get end time
    double timeParallel = endTimePar - startTimePar;

    // --- Print Results ---
    printf("\nSorted (Sequential):\n");
    printArray(arraySequential, n);
    printf("Time (Sequential Merge Sort): %.6f s\n", timeSequential);

    printf("\nSorted (Parallel):\n");
    printArray(arrayParallel, n);
    printf("Time (Parallel Merge Sort):   %.6f s\n", timeParallel);

    // Calculate and print speedup
    if (timeParallel > 1e-9 && timeSequential > 1e-9) { // Avoid division by zero or tiny numbers
         printf("\n%-25s %.2fx speedup\n", "Speedup (Par / Seq):", timeSequential / timeParallel);
    } else {
         printf("\nSpeedup calculation skipped due to very small timings.\n");
    }


    // --- Cleanup ---
    free(arraySequential);
    free(arrayParallel);
    return 0;
}
