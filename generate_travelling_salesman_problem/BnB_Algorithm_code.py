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

# dist_matrix =   [[0,5,0,6,5,4],
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
    print("min_cost: ", min_cost)
    print("min_path: ", min_path)

def initial(dist_matrix):
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix[0])):
            if i == j:
                dist_matrix[i][j] = float('inf')
            if dist_matrix[i][j] == 0:
                dist_matrix[i][j] = float('inf')
    return dist_matrix



def branch_and_bound_dfs(dist_matrix, visited, path):
    # Not sure if dist_matrix reassignment is needed
    root_matrix = copy_matrix(dist_matrix)
    root_matrix = initial(root_matrix)
    row_cost = matrix_reduction_by_axis(root_matrix, None, 'row')
    col_cost = matrix_reduction_by_axis(root_matrix, None, 'col')
    # print("root matrix: \n", root_matrix)
    # print(row_cost + col_cost)
    for i in range(len(dist_matrix)):
        visited.add(i)
        path.append(i)
        search(root_matrix, visited, path, row_cost + col_cost, i)
        visited.remove(i)
        path.remove(i)
    
def calculate_actual_cost(path):
    cost = 0
    global dist_matrix
    # print("distance matrix in calculated: ", dist_matrix)
    for i in range(len(path)):
        cost += dist_matrix[path[-1 + i]][path[i]]
        # print(path[-1+i], path[i])
    return cost
    
def search(curr_matrix, visited, path, curr_cost, curr_node):
    # print(path)
    # base case
    global min_cost
    global min_path
    global dist_matrix
    # print(path)
    if len(visited) == len(curr_matrix):
        # + dist_matrix[path[-1]][path[0]]:
        # + curr_matrix[path[-1]][path[0]]
        # curr_cost = calculate_actual_cost(path)
        actual_curr_cost = calculate_actual_cost(path)
        # print(path, curr_cost)
        if min_cost >= actual_curr_cost:
            min_cost = actual_curr_cost
            min_path = path[:]
            print("path found: ", path, min_cost)
        return

    actual_curr_cost = calculate_actual_cost(path)
    # print("actual_curr_cost: ", actual_curr_cost)
    # print("min_cost", min_cost)
    if actual_curr_cost > min_cost:
        # print("entered here")
        return
    
    pq = []
    for idx in range(len(curr_matrix)):
        if idx not in visited:
            new_matrix, new_cost = examine_child_node(curr_matrix, curr_cost, curr_node, idx)
            #  order converted for heapq in ascending order
            # print("new_cost: ", new_cost)
            # print("new_matrix: \n", new_matrix)
            # print(type(new_matrix))
            # print("idx", idx)

            heapq.heappush(pq, (new_cost, idx, new_matrix))

    

    while pq:
        # print("heap pq: ", pq)
        new_cost, node, new_matrix = heapq.heappop(pq)
        # print("node", node)
        # print("node explored order: ", node)
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
            # print("segment: \n", segment)
            min_elt = min(segment)
            # print("min_elt: \n", min_elt)
            if min_elt > 0:
                #  reduce for each_row or each_col
                # print("entered")
                for j in range(len(segment)):
                    segment[j] -= min_elt
                # print("segment after calc: \n", segment)
                cost_reduc += min_elt
    # print("matrix reduction: \n", matrix)
    # print("cost_reduc: ", cost_reduc)
    return cost_reduc

def examine_child_node(curr_matrix, curr_cost, parent_node, curr_node):
    new_matrix = copy_matrix(curr_matrix)
    cost_reduc = 0
    new_matrix[parent_node, :] = np.inf
    new_matrix[:, curr_node] = np.inf
    new_matrix[parent_node][curr_node] = np.inf
    # print("new_matrix: \n", new_matrix)
    cost_reduc += matrix_reduction_by_axis(new_matrix, parent_node, 'row')
    cost_reduc += matrix_reduction_by_axis(new_matrix, curr_node, 'col')
    # print("examine chilld node: ", curr_matrix[parent_node][curr_node])
    return [new_matrix, curr_cost + curr_matrix[parent_node][curr_node] + cost_reduc]


def copy_matrix(original_matrix):
    new_matrix = np.array([row[:] for row in original_matrix], dtype=float)
    # print("new_matrix: \n", new_matrix)
    return new_matrix


main()




