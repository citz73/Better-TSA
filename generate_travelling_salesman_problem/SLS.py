import time
import numpy as np
import random

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
        lines = f.read()
        
    with open(file_name) as f:
        line = f.readline()
        
    num_cities = int(line)
    temp = lines[1:][1:].split('\n')
    temp = list(filter(None, temp))
    dist_matrix = []
    for n in range(num_cities):
        dist_matrix.append(list(np.float_(temp[n].split(' '))))
        
    return num_cities, dist_matrix
        
def calculate_cost(dist_matrix, current):
    cost = 0
    for i in range(len(current)):
        if i == len(current) - 1:
            cost = cost + dist_matrix[current[i]][current[0]]
        else:
            src = current[i]
            dest = current[i + 1]
            cost = cost + dist_matrix[src][dest]
    return cost

def generate_neighbors(current, visited):
    neighbors = []
    for i in range(1,len(current)):
        for j in range(1, len(current)):
            if i < j:
               neighbor = current.copy()
               neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
               if neighbor not in visited:
                   neighbors.append(neighbor)
    return neighbors
    
def stochastic_local_search(num_cities, dist_matrix, start):  
    current = list(np.random.permutation(num_cities)) # initialize current state
    visited = [current]
    greedy_rate = 0.5
    time_limit = 10
    
    time_elapsed = 0
    while time_elapsed < time_limit:
        current_cost = calculate_cost(dist_matrix, current)
        neighbors = generate_neighbors(current, visited)
        if len(neighbors) == 0:
            current.append(current[0])
            return current, current_cost
        
        if np.random.uniform(0,1) < greedy_rate: # take greedy step
            for n in range(len(neighbors)):
                new = neighbors[n]
                new_cost = calculate_cost(dist_matrix, new)
                if current_cost > new_cost:
                    current = new
        else: # take random step
            current = random.choice(neighbors)
            
        if current not in visited:
            visited.append(current)
        time_elapsed = time.time() - start
        
    current.append(current[0])
    return current, current_cost

if __name__ == '__main__':
    
    # n = int(input("Enter the number of locations: "))
    # mean = float(input("Enter the mean: "))
    # sigma = float(input("Enter the standard deviation: "))

    # write_distance_matrix(n, mean, sigma)
    
    start = time.time()
    file_name = '10_0.0_1.0.out'
    num_cities, dist_matrix = read_input(file_name)
    path, cost = stochastic_local_search(num_cities, dist_matrix, start)
    print(path, cost)
    
    