import heapq
import numpy as np
import math

min_cost = np.inf
min_path = []
# dist_matrix =     [[0, 10, 15, 20],
#                    [10, 0, 35, 25],
#                    [15, 35, 0, 30],
#                    [20, 25, 30, 0]]


dist_matrix =      [[0,67,46,55,75], 
                    [67,0,91,122,153],
                    [46,91,0,98,65],
                    [55,122,98,0,40],
                    [75,153,65,40,0]]

# dist_matrix =  [[0,5,0,6,5,4],
#                 [5,0,2,4,3,0],
#                 [0,2,0,1,0,0],
#                 [6,4,1,0,7,0],
#                 [5,3,0,7,0,3],
#                 [4,0,0,0,3,0]]

def main():
    visited, curr_cost, parent = set(), 0, 0
    path = []
    global min_cost
    global min_path
    branch_and_bound_dfs(dist_matrix, visited, path)
    print("min_cost", min_cost) 
    print("min_path", min_path)

def initial(dist_matrix):
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix[0])):
            if i == j:
                dist_matrix[i][j] = float('inf')
            if dist_matrix[i][j] == 0:
                dist_matrix[i][j] = float('inf')
    return dist_matrix

def branch_and_bound_dfs(dist_matrix, visited, path):
    root_matrix = copy_matrix(dist_matrix)
    root_matrix = initial(root_matrix)
    row_cost = matrix_reduction_by_axis(root_matrix, None, 'row')
    col_cost = matrix_reduction_by_axis(root_matrix, None, 'col')
    for i in range(len(dist_matrix)):
        visited.add(i)
        path.append(i)
        search(root_matrix, visited, path, row_cost + col_cost, i)
        visited.remove(i)
        path.remove(i)
    
def calculate_actual_cost(path):
    cost = 0
    global dist_matrix
    for i in range(len(path) - 1):
        cost += dist_matrix[path[i]][path[i+1]]
    return cost
    
def search(curr_matrix, visited, path, curr_cost, curr_node):
    global min_cost
    global min_path
    global dist_matrix
    if len(visited) == len(curr_matrix):
        actual_curr_cost = calculate_actual_cost(path)
        if min_cost > actual_curr_cost + dist_matrix[path[-1]][path[0]]:
            min_cost = actual_curr_cost + dist_matrix[path[-1]][path[0]]
            min_path = path[:]
        return

    actual_curr_cost = calculate_actual_cost(path)
    if actual_curr_cost > min_cost:
        return
    
    pq = []
    for idx in range(len(curr_matrix)):
        if idx not in visited:
            new_matrix, new_cost = examine_child_node(curr_matrix, curr_cost, curr_node, idx)
            heapq.heappush(pq, (new_cost, idx, new_matrix))

    while pq:
        new_cost, node, new_matrix = heapq.heappop(pq)
        visited.add(node)
        path.append(node)
        search(curr_matrix, visited, path, new_cost, node)
        visited.remove(node)
        path.pop()

def matrix_reduction_by_axis(curr_matrix, node, axis):
    cost_reduc = 0
    matrix = None
    if axis == 'row':
        matrix = curr_matrix
    else:
        matrix = curr_matrix.transpose()
    for row in range(len(matrix)):
        if row is not node:
            segment = matrix[row]
            min_elt = min(segment)
            if min_elt > 0:
                for j in range(len(segment)):
                    segment[j] -= min_elt
                cost_reduc += min_elt
    return cost_reduc

def examine_child_node(curr_matrix, curr_cost, parent_node, curr_node):
    new_matrix = copy_matrix(curr_matrix)
    cost_reduc = 0
    new_matrix[parent_node, :] = np.inf
    new_matrix[:, curr_node] = np.inf
    new_matrix[parent_node][curr_node] = np.inf
    cost_reduc += matrix_reduction_by_axis(new_matrix, parent_node, 'row')
    cost_reduc += matrix_reduction_by_axis(new_matrix, curr_node, 'col')
    return [new_matrix, curr_cost + curr_matrix[parent_node][curr_node] + cost_reduc]


def copy_matrix(original_matrix):
    new_matrix = np.array([row[:] for row in original_matrix], dtype=float)
    return new_matrix


main()




