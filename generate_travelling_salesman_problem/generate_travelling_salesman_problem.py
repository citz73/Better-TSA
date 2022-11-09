import numpy as np

def write_distance_matrix(n, mean, sigma):
    distance_matrix = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            distance = 0
            while distance <= 0:
                distance = np.random.normal(mean, sigma)
                distance_matrix[row][col] = distance

    np.savetxt(
        f"{n}_{mean}_{sigma}.out",
        distance_matrix,
        delimiter=" ",
        fmt="%1.4f",
        header=str(n),
        comments="",
    )
    return distance_matrix

#  Every road is connected
#  thus it should be n by n

min_cost = float("inf")
min_path = []

def naiveTSP(distance_matrix, visited, curr_cost, parent, path):
    if len(visited) == n:
        global min_cost
        if min_cost >= curr_cost + distance_matrix[parent][path[0]] - distance_matrix[0][path[0]]:
            min_cost = curr_cost + distance_matrix[parent][path[0]] - distance_matrix[0][path[0]]
            min_path.clear()
            min_path.append(path[:])
        return
    
    
    if curr_cost > min_cost:
        return

    for i in range(n):
        if i not in visited:
            visited.add(i)
            path.append(i)
            naiveTSP(distance_matrix, visited, curr_cost + distance_matrix[parent][i], i, path)
            path.pop()
            visited.remove(i)
    return 



# if __name__ == "__main__":
# n = int(input("Enter the number of locations: "))
# mean = float(input("Enter the mean: "))
# sigma = float(input("Enter the standard deviation: "))
visited, curr_cost, parent = set(), 0, 0
# distance_matrix = [[0, 10, 15, 20],
#                    [10, 0, 35, 25],
#                    [15, 35, 0, 30],
#                    [20, 25, 30, 0]]

distance_matrix = [ [0,67,46,55,75], 
                    [67,0,91,122,153],
                    [46,91,0,98,65],
                    [55,122,98,0,40],
                    [75,153,65,40,0]]

n = len(distance_matrix)
print(n)
path = []

naiveTSP(distance_matrix, visited, curr_cost, parent, path)

print(min_path, min_cost)