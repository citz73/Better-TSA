import heapq
import numpy as np
import math
import sys

min_cost = np.inf
min_path = []

def write_distance_matrix(n, mean, sigma):
    distance_matrix = np.zeros((n, n))
    random_distance = []
    num_distance = int(n * (n-1) / 2)
    for _ in range(num_distance):
        distance = 0
        while distance <= 0:
            distance = np.random.normal(mean, sigma)

        random_distance.append(distance)
    
    iu = np.triu_indices(n, 1)
    distance_matrix[iu] = random_distance
    distance_matrix += distance_matrix.T

    np.savetxt(
        f"{n}_{mean}_{sigma}.out",
        distance_matrix,
        delimiter=" ",
        fmt="%1.4f",
        header=str(n),
        comments="",
    )

def read_input(file_name):
    
    with open(file_name) as f:
        num_cities = int(f.readline())
        lines = f.read()
        
    temp = lines.split('\n')
    temp = list(filter(None, temp))
    dist_matrix = []
    for n in range(num_cities):
        dist_matrix.append(list(np.float_(temp[n].split(' '))))
        
    return num_cities, dist_matrix

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
        search(dist_matrix, root_matrix, visited, path, row_cost + col_cost, i)
        visited.remove(i)
        path.remove(i)
    
def calculate_actual_cost(path, dist_matrix):
    cost = 0
    for i in range(len(path) - 1):
        cost += dist_matrix[path[i]][path[i+1]]
    return cost
    
def search(dist_matrix, curr_matrix, visited, path, curr_cost, curr_node):
    global min_cost
    global min_path
    if len(visited) == len(curr_matrix):
        actual_curr_cost = calculate_actual_cost(path, dist_matrix)
        if min_cost > actual_curr_cost + dist_matrix[path[-1]][path[0]]:
            min_cost = actual_curr_cost + dist_matrix[path[-1]][path[0]]
            min_path = path[:]
        return

    actual_curr_cost = calculate_actual_cost(path, dist_matrix)
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
        search(dist_matrix, curr_matrix, visited, path, new_cost, node)
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

def main():
    
    mode = sys.argv[1]
    # mode = input()
    
    if mode == 'g': # generate new distance matrix
        n = int(input("Enter the number of locations: "))
        mean = float(input("Enter the mean: "))
        sigma = float(input("Enter the standard deviation: "))
        write_distance_matrix(n, mean, sigma)
        
        file_name = str(n) + '_' + str(mean) + '_' + str(sigma) + '.out'
        
    elif mode == 'e' : # use existing file
        n = int(input("Enter the number of locations: "))
        mean = float(input("Enter the mean: "))
        sigma = float(input("Enter the standard deviation: "))
        
        file_name = str(n) + '_' + str(mean) + '_' + str(sigma) + '.out'
    
    num_cities, dist_matrix = read_input(file_name)
    
    visited, curr_cost, parent = set(), 0, 0
    path = []
    global min_cost
    global min_path
    
    branch_and_bound_dfs(dist_matrix, visited, path)
    min_path.append(min_path[0])
    
    print('\n**************')
    print("*** Path founded:" + str(min_path))
    print("*** Cost: " + str(round(min_cost,4)))
    print('**************\n')

if __name__ == '__main__':    
    main()



