import numpy as np
from scipy.optimize import linear_sum_assignment
import time

num_jobs = 1000
num_workers = 1000

def hungarian_algorithm(cost_matrix):
    start_time = time.time()
    # Ensure that the cost matrix is square by padding with zeros if necessary
    n, m = cost_matrix.shape
    if n > m:
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, n - m)), mode='constant')

    # Find the optimal assignment using the linear_sum_assignment function
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time used to find the total cost: {total_time} seconds")

    return row_ind, col_ind

# Example usage:
cost_matrix = np.random.randint(1, 100, size=(num_workers, num_jobs))

row_ind, col_ind = hungarian_algorithm(cost_matrix)
print("Optimal Assignment:")
for i in range(len(row_ind)):
    print(f"Task {i} is assigned to Worker {col_ind[i]}")
