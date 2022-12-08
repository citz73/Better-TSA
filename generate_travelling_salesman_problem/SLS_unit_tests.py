import time
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

from SLS import stochastic_local_search


def create_distance_matrix(n, mean=0, sigma=1): 
    return np.abs(np.random.normal(mean, sigma, size=(n,n)))


def get_stats(costs):
    ''' analyze ensemble '''

    min_cost = np.min(costs)
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)

    print('minimum: ', np.round(min_cost, 2))
    print('average: ', np.round(avg_cost, 2))
    print(f'stddev: ', np.round(std_cost, 2))

    return min_cost, avg_cost, std_cost


def run_ensemble(n, num_iter, greedy_rate=0.5, d=None): 

    print(f'\nstarting ensemble (n={n})... ')

    if d is None: 
        d = create_distance_matrix(n)

    paths, costs = [], []
    for _ in tqdm(range(num_iter)): 
        path, cost = stochastic_local_search(n, d, time.time(), greedy_rate)
        costs.append(cost)

    return paths, costs
    

def test0(): 
    ''' shows variation for different trials ''' 

    n = 5
    d = create_distance_matrix(n)

    costs = []
    runs = 10
    for _ in range(runs): 
        _, cost = run_ensemble(n, 1, d=d)
        costs.append(cost[0])

    plt.clf(), plt.close()
    plt.figure()
    plt.plot(costs, 'o')
    plt.ylim([0, np.max(costs)+3])
    plt.xlabel('trial number'), plt.ylabel('cost')
    plt.title(f'variation in cost of TSP across different trials\n(n = {n})')
    plt.savefig('../figures/test0.png')
    # plt.show()
    print('figure saved!')


def test1():
    ''' tests variation across different trials with the same distance matrix '''

    n = 5
    # num_iters = [10, 40, 70, 100]
    num_iters = [100]*4
    d = create_distance_matrix(n)
    
    min_costs, min_cost_idxs = [], {}
    plt.clf(), plt.close()
    plt.subplots(2,2, figsize=(11,6), layout='constrained')
    for i, num_iter in enumerate(num_iters):
        _, costs = run_ensemble(n, num_iter, d=d)

        # get stats
        min_cost, avg_cost, std_cost = get_stats(costs)

        # calculate minimum cost (= most optimal value)
        min_costs.append(min_cost)
        min_cost_idxs[i] = np.where(np.array(costs)==min_cost)[0]

        plt_idx = 4 if i==2 else 5 if i==3 else i+1
        plt.subplot(2,3,plt_idx)
        plt.plot(costs, 'o', label='suboptimal costs')
        plt.plot(min_cost_idxs[i], [min_cost]*len(min_cost_idxs[i]), 'ro', label='optimal costs')
        plt.axhline(avg_cost, color='y', linestyle='dashed')
        plt.axhline(avg_cost+std_cost, color='y', linestyle='dotted')
        plt.axhline(avg_cost-std_cost, color='y', linestyle='dotted')
        # plt.ylim([0,np.max(costs)+np.min(costs)])
        plt.ylim([0,10])
        plt.xlabel('iteration'), plt.ylabel('cost')
        plt.title(f'num iters = {num_iter}')

    # print(f'\nfirst optimal solution reached at: ')
    min_cost_idxs_first = []
    for key in min_cost_idxs: 
        min_cost_idxs_first.append(min_cost_idxs[key][0])

    plt.subplot(3,4,8), plt.plot(range(1,5), min_cost_idxs_first, 'o')
    plt.xlabel('ensemble #'), plt.ylabel('epoch of optimal cost')
    plt.title('number of iterations to find\noptimal cost')

    # plt.legend()
    # plt.show()
    plt.savefig('../figures/test1.png')
    print('figure saved!')


def test2():
    ''' how increasing the size of the distance matrix affects finding optimal solution ''' 

    ns = [5, 6, 7, 8]
    # ns = [5]*4
    num_iter = 20

    min_costs, min_cost_idxs = [], {}
    plt.clf(), plt.close()
    plt.subplots(2,2, figsize=(11,6), layout='constrained')
    for i, n in enumerate(ns): 
        _, costs = run_ensemble(n, num_iter)

        # get stats
        min_cost, avg_cost, std_cost = get_stats(costs)

        # calculate minimum cost (= most optimal value)
        min_costs.append(min_cost)
        min_cost_idxs[n] = np.where(np.array(costs)==min_cost)

        plt_idx = 4 if i==2 else 5 if i==3 else i+1
        plt.subplot(2,3,plt_idx)
        plt.plot(costs, 'o', label='suboptimal costs')
        plt.plot(min_cost_idxs[n], [min_cost]*len(min_cost_idxs[n]), 'ro', label='optimal costs')
        plt.axhline(avg_cost, color='y', linestyle='dashed')
        plt.axhline(avg_cost+std_cost, color='y', linestyle='dotted')
        plt.axhline(avg_cost-std_cost, color='y', linestyle='dotted')
        # plt.ylim([0,np.max(costs)+np.min(costs)])
        plt.ylim([0,12])
        plt.xlabel('iteration'), plt.ylabel('cost')
        plt.title(f'n={n}')
        # plt.legend()

    plt.subplot(3,4,8), plt.plot(ns, min_costs, 'o')
    plt.xlabel('number of nodes'), plt.ylabel('optimal cost')
    plt.title('number of nodes\nvs. optimal cost')

    # plt.show()
    plt.savefig('../figures/test2.png')
    print('figure saved!')


def test3(): 
    ''' affect of greedy rate on finding optimal solution '''

    n = 6
    num_iter = 50
    greedy_rates = [0.5, 0.6, 0.7, 0.8]
    d = create_distance_matrix(n)

    min_costs, min_cost_idxs = [], {}
    plt.clf(), plt.close()
    plt.subplots(2,2, figsize=(11,6), layout='constrained')
    for i, greedy_rate in enumerate(greedy_rates): 
        _, costs = run_ensemble(n, num_iter, greedy_rate=greedy_rate, d=d)

        # get stats
        min_cost, avg_cost, std_cost = get_stats(costs)

        # calculate minimum cost (= most optimal value)
        min_costs.append(min_cost)
        min_cost_idxs[n] = np.where(np.array(costs)==min_cost)

        plt_idx = 4 if i==2 else 5 if i==3 else i+1
        plt.subplot(2,3,plt_idx)
        plt.plot(costs, 'o', label='suboptimal costs')
        plt.plot(min_cost_idxs[n], [min_cost]*len(min_cost_idxs[n]), 'ro', label='optimal costs')
        plt.axhline(avg_cost, color='y', linestyle='dashed')
        plt.axhline(avg_cost+std_cost, color='y', linestyle='dotted')
        plt.axhline(avg_cost-std_cost, color='y', linestyle='dotted')
        # plt.ylim([0,np.max(costs)+np.min(costs)])
        plt.ylim([0,10])
        plt.xlabel('iteration'), plt.ylabel('cost')
        plt.title(f'greedy rate = {greedy_rate}')
        # plt.legend()

    plt.subplot(3,4,8), plt.plot(greedy_rates, min_costs, 'o')
    plt.xlabel('greedy rate'), plt.ylabel('optimal cost')
    plt.title('greedy rate\nvs. optimal cost')

    # plt.show()
    plt.savefig('../figures/test3.png')
    print('figure saved!')


if __name__ == '__main__':

    import sys 
    mode = sys.argv[1]

    if mode == '0': 
        print('running test 0')
        test0()

    if mode == '1': 
        print('running test 1')
        test1()

    elif mode == '2': 
        print('running test 2')
        test2()

    elif mode == '3':
        print('running test 3')
        test3()