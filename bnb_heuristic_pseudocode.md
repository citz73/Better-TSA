# pseudocode for BNB-DFS heuristic

def row_col_reduction(m): 
	‘’’ reduces each row of an nxn matrix m by the smallest value of the row, then reduces each column of m by the smallest value of the column.’’’
	
	total_cost = 0

	for row in m: 
		m[row,:] = m[row,:] - min(m[row,:])
		total_cost += min(m[row,:])
	for col in m: 
		m[:,col] = m[:,col] - min(m[:,col])
		total_cost += min(m[:,col])

	return m, total_cost

def select_next_node(unexplored_node_indices): 
	''' given a list of unexplored nodes, compute the costs and select the next node to expand. '''

	cost_cur = infinity

	for j in list_of_remaining_node_indices: 
		m_prev = replace_elements_with_infinity(m_prev[i,:], m_prev[:,j], m_prev[j,i])
		m_prev, reduction_cost = row_col_reduction(m_prev)
		cost_total = cost_prev + m_prev[i,j] + reduction_cost
		if cost_total < cost_cur: 
			node_next = j
			m_next = m_prev
			cost_next = cost_total

	return node_next, m_next, cost_next

# import distance matrix
distance_matrix = some symmetric matrix with infinity diagonal elements

# row and column reduce the raw distance matrix
m_root, cost_root = row_col_reduction(distance_matrix)

# find 
explored_node_indices = [root_node_index]
unexplored_node_indices = delete root_node_index from [all_indices]
i = root_node_index
m_prev = m_root
cost_prev = cost_root

# compute the costs of the remaining nodes
while unexplored_node_indices: 
	i, m_prev, cost_prev = select_next_node(unexplored_node_indices, i, m_prev, cost_prev)
	explored_node_indices = add i to explored_node_indices
	unexplored_node_indices = delete i from unexplored_node_indices

# result
explored_node_indices # order of the nodes expanded