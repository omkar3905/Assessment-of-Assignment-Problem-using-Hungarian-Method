import numpy as np
import random

# Define the cost matrix
cost_matrix = np.array([
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
])

assigned_jobs = []

# Step 1: For each row, find the smallest element and subtract it from every element in its row
for i in range(cost_matrix.shape[0]):
    min_val = np.min(cost_matrix[i])
    cost_matrix[i] -= min_val

# Step 2: For each column, find the smallest element and subtract it from every element in its column
for j in range(cost_matrix.shape[1]):
    min_val = np.min(cost_matrix[:, j])
    cost_matrix[:, j] -= min_val

# Step 3: Assign the job randomly to the workers for columns with only one zero
for j in range(cost_matrix.shape[1]):
    if np.count_nonzero(cost_matrix[:, j] == 0) == 1:
        row_index = np.where(cost_matrix[:, j] == 0)[0][0]
        assigned_jobs.append(j)
        print(f"Assigning Job {j} to Worker {row_index}")

# Print the updated cost matrix after steps 1 and 2
print("Cost Matrix after Steps 1 and 2:")
print(cost_matrix)

# Print the cost matrix as an array
print("\nCost Matrix as an Array:")
print(np.array_str(cost_matrix, precision=2, suppress_small=True))

# Creating a list where the index corresponds to the workers and the job value at that index
for i in range(cost_matrix.shape[0]):
    zero_indices = np.where(cost_matrix[i] == 0)[0]
    available_jobs = [job for job in range(cost_matrix.shape[1]) if job not in assigned_jobs]
    job_assigned = next((job for job in zero_indices if job not in assigned_jobs), None)
    if job_assigned is not None:
        assigned_jobs.append(job_assigned)
    else:
        if available_jobs:
            random_job = random.choice(available_jobs)
            assigned_jobs.append(random_job)

# Print the list
print("\nList where index corresponds to workers and job value at that index:")
print(assigned_jobs)
