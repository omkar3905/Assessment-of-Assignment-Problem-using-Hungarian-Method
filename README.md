
# Assessment of Assignment Problem using Hungarian Method

This repository contains the implementation and analysis of the Hungarian Method to solve the Assignment Problem. The assignment problem involves determining the most efficient way to distribute a set of resources to a set of tasks, minimizing the total cost or maximizing the total profit. The Hungarian Method is known for providing optimal solutions efficiently.


## Table of Contents

1. Introduction

2. Literature Survey

3. Methodology

4. Conclusion

5. Future Work

6. References
## Introduction

This study presents a detailed case analysis of the Hungarian Method for solving the Assignment Problem. The computational outcomes demonstrate the method's effectiveness in providing optimal solutions across various scenarios. The findings are intended to aid decision-makers in mitigating production-related risks and adapting to sustainable market changes.
## Literature Survey

The Hungarian Method has been extensively used and modified to address both balanced and unbalanced assignment problems. Various researchers have proposed enhancements and alternative algorithms to improve the efficiency and applicability of the Hungarian Method. This section summarizes the contributions of notable works in the field, including approaches using genetic algorithms, ant colony optimization, and other heuristic methods.
## Methodology

➤ Assumptions

 • The number of assignees and the number of tasks are equal.

 • Each assignee is assigned exactly one task.

 • Each task is performed by exactly one assignee.

 • A cost 𝐶𝑖𝑗 is associated with assignee 𝑖 performing task 𝑗.

 • The objective is to minimize the total cost.

➤ Balanced Assignment Problem

For balanced assignment problems, where the number of assignees equals the number of tasks, the following steps are used:

1. Subtract the minimum element of each row from all elements of that row.

2. Subtract the minimum element of each column from all elements of that column.

3. Cover all zeros in the resulting matrix using the minimum number of horizontal and vertical lines.

4. Adjust the matrix and repeat until an optimal assignment is found.

➤ Unbalanced Assignment Problem

For unbalanced assignment problems, dummy tasks or assignees are added to balance the matrix. The Hungarian Method is then applied similarly to the balanced problem.
## Conclusion

The Hungarian Method proves to be an effective solution for both balanced and unbalanced assignment problems. It offers an optimal solution that minimizes the total cost and handles complex scenarios efficiently.
## Future Work

Future research may explore:

 • Enhancements to the Hungarian Method for large-scale problems.

 • Integration with other optimization algorithms.
 
 • Application to real-world scenarios in various industries.
## References

 • Kuhn, H. W. (1955). The Hungarian method for the assignment problem.

 • Chopra, S., et al. (2017). A comparative study of assignment algorithms.

 • Rabbani, M., Khan, M., & Quddoos, M. (2019). Modified Hungarian Method for unbalanced assignment problems.

 • Kumar, A. (2006). Solving unbalanced assignment problems.

 • Xiang, S., & Liu, Y. (2021). Integrating berth allocation and quay crane assignment problem.