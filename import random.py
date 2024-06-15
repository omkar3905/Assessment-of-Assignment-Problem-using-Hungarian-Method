import random
import numpy as np
import time
from ortools.linear_solver import pywraplp
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import multiprocessing

# Define the parameters
population_size = 1500
mutation_rate = 0.5
max_generations = 9999999999999999
no_improvement_limit = 500

# Create a function to solve the assignment problem using a genetic algorithm
def solve_assignment_problem_genetic_parallel(cost_matrix, num_jobs, num_workers, population_size, mutation_rate, max_generations, no_improvement_limit):
    start_time = time.time()

    def create_individual():
        return random.sample(range(num_jobs), num_jobs)

    def calculate_fitness(individual):
        total_cost = 0
        for worker, job in enumerate(individual):
            total_cost += cost_matrix[worker][job]
        return 1 / (total_cost + 1), total_cost

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, num_jobs - 1)
        child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            swap_indices = random.sample(range(num_jobs), 2)
            individual[swap_indices[0]], individual[swap_indices[1]] = individual[swap_indices[1]], individual[swap_indices[0]]
        return individual

    def evaluate_fitness(individual):
        return calculate_fitness(individual)

    def parallel_fitness_evaluation(population):
        with multiprocessing.Pool(processes=4) as pool:  # Adjust the number of processes as needed
            fitness_values = pool.map(evaluate_fitness, population)
        return fitness_values

    # Initial population
    population = [create_individual() for _ in range(population_size)]

    generation = 0
    best_fitness = 0
    generations_without_improvement = 0

    while generation < max_generations and generations_without_improvement < no_improvement_limit:
        # Parallel fitness evaluation
        fitness_values = parallel_fitness_evaluation(population)
        fitness_values = list(zip(fitness_values, population))
        fitness_values.sort(reverse=True)  # Sort by fitness value

        # Select parents
        num_parents = int(population_size * 0.2)
        parents = [individual for fitness, individual in fitness_values[:num_parents]]
        next_generation = parents

        # Crossover and mutation
        while len(next_generation) < population_size:
            parent1, parent2 = random.choices(parents, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

        current_best_fitness, best_individual = fitness_values[0]
        if current_best_fitness[0] > best_fitness:
            best_fitness = current_best_fitness[0]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        generation += 1
        print(f"Generation {generation} - Best Fitness: {best_fitness}")

    best_individual = max(fitness_values, key=lambda x: x[0][0])[1]
    best_fitness, best_cost = calculate_fitness(best_individual)

    print(f"Optimal Solution Found in Generation {generation}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken to find the optimal solution: {total_time} seconds")

    print(f"Best Assignment (Genetic Algorithm): {best_individual}, Best Fitness: {best_fitness}, best cost: {best_cost}")

    return best_individual, best_fitness, best_cost

# ... (rest of the code, including the test case and Hungarian integration)
def print_matrix(matrix):
    for row in matrix:
        print(row)

# Test case
test_case = {
    "name": "Test Case 1",
    "cost_matrix": np.random.randint(1, 100, size=(50,50))
}

t = test_case["cost_matrix"]
num_jobs = num_workers = len(t)

print("Randomly created cost matrix for genetic algorithm:")
print_matrix(test_case["cost_matrix"])

# Use the Hungarian algorithm for initial assignment
row_ind, col_ind = linear_sum_assignment(t)
t = t[row_ind][:, col_ind]
num_jobs = num_workers = len(t)
print(len(t))

cost_matrix = t
print(cost_matrix)
print('---')
print(t)

best_assignment_genetic, best_fitness_genetic, best_cost_genetic = solve_assignment_problem_genetic(t, num_jobs, num_workers, population_size, mutation_rate, max_generations, no_improvement_limit)

print(f"{test_case['name']} - Best Assignment (Genetic Algorithm): {best_assignment_genetic}, Best Fitness: {best_fitness_genetic}, best cost: {best_cost_genetic}")

# Integrate the Hungarian algorithm
def solve_assignment_problem_hungarian(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost

row_ind, col_ind, total_cost_hungarian = solve_assignment_problem_hungarian(test_case["cost_matrix"])

# Create an assignment matrix based on Hungarian results
assignment_matrix = np.zeros_like(test_case["cost_matrix"], dtype=int)
assignment_matrix[row_ind, col_ind] = 1

print(f"Total cost (Hungarian Algorithm): {total_cost_hungarian}")

print("Assignment Matrix (Hungarian Algorithm):")
print(assignment_matrix)

# Visualize the assignment using a heatmap
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(test_case["cost_matrix"], cmap='viridis')
plt.title("Cost Matrix")
plt.colorbar()
for i, j in enumerate(col_ind):
    plt.text(j, i, 'X', color='red', fontsize=12, ha='center', va='center')

plt.subplot(132)
plt.imshow(assignment_matrix, cmap='viridis')
plt.title("Assignment Matrix (Hungarian)")
plt.colorbar()

plt.subplot(133)
plt.imshow(test_case["cost_matrix"] * assignment_matrix, cmap='viridis')
plt.title("Cost Matrix with Assignment (Hungarian)")
plt.colorbar()
plt.show()
def main():
    # Data
    costs = t
    num_workers = len(t)
    num_tasks = len(t[0])

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total cost from the linear programming model: {solver.Objective().Value()}\n")
        for i in range(num_workers):
            for j in range(num_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    print(f"Worker {i} assigned to task {j} with cost: {costs[i][j]}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()