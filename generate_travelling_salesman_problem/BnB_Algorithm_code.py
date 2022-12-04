import heapq

min_cost = float('inf')
min_path = []

def main():
    dist_matrix = []
    visited = set()
    parent_node = -1
    path = []
    branch_and_bound_dfs(dist_matrix, visited, parent_node, path)

def branch_and_bound_dfs(dist_matrix, visited, parent_node, path):
    # Not sure if dist_matrix reassignment is needed
    dist_matrix = [[float('inf') if i == j else j for j in range(len(dist_matrix[0]))] for i in range(len(dist_matrix))]
    root_matrix = copy_matrix(dist_matrix)
    row_cost = matrix_reduction_by_axis(root_matrix, None, 'row')
    col_cost = matrix_reduction_by_axis(root_matrix, None, 'col')
    search(root_matrix, visited, path, row_cost + col_cost, None, parent_node)

def search(curr_matrix, visited, path, curr_cost, curr_node, parent_node):
    # base case
    if len(visited) == len(curr_matrix):
        if min_cost > curr_cost + curr_matrix[path[-1]][path[0]]:
            global min_cost
            global min_path
            min_cost = curr_cost + curr_matrix[path[-1]][path[0]]
            min_path = path
        return

    if curr_cost > min_cost:
        return
    
    pq = []
    for idx in range(len(curr_matrix)):
        if idx not in visited:
            new_matrix, new_cost = examine_child_node(curr_matrix, curr_cost, curr_node, idx)
            #  order converted for heapq in ascending order
            heapq.heappush(pq, (new_cost, new_matrix, idx))

        while pq:
            new_matrix, new_cost, node = heapq.heappop(pq)
            visited.add(node)
            path.append(node)
            search(curr_matrix, visited, path, new_cost, node, curr_node)
            visited.remove(node)
            path.pop()

def matrix_reduction_by_axis(curr_matrix, node, axis):
    cost_reduc = 0
    matrix = None
    if axis == 'row':
        matrix = curr_matrix
    else:
        matrix = curr_matrix.transpose()
    for row in len(matrix):
        if row is not node:
            segment = curr_matrix[row]
            min_elt = min(segment)
            if min_elt > 0:
                #  reduce for each_row or each_col
                for j in segment:
                    j -= min_elt
                cost_reduc += min_elt
    return cost_reduc

def examine_child_node(curr_matrix, curr_cost, parent_node, curr_node):
    new_matrix = copy_matrix(curr_matrix)
    cost_reduc = 0
    new_matrix[parent_node, :] = float("inf")
    new_matrix[:, curr_node] = float("inf")
    new_matrix[curr_node, parent_node] = float("inf")
    cost_reduc += matrix_reduction_by_axis(new_matrix, parent_node, 'row')
    cost_reduc += matrix_reduction_by_axis(new_matrix, curr_node, 'col')
    return [new_matrix, curr_cost + curr_matrix[parent_node][curr_node] + cost_reduc]


def copy_matrix(original_matrix):
    new_matrix = [row[:] for row in range(len(original_matrix))]
    return new_matrix