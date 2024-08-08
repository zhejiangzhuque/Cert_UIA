import numpy as np

def dfs(data):
    def _dfs(u):
        conns.append(u)
        neigbrs = edge_index[edge_index[:,0]==u]
        for v in neigbrs[:,1]:
            if not visit[v]:
                visit[v] = True
                _dfs(v)

    edge_index = data.edge_index.t().data.cpu().numpy()
    N = data.num_nodes
    visit = [False] * N
    s = edge_index[0,0]
    conns = []
    visit[s] = True
    _dfs(s)
    
    return sum(visit)==N, np.array(conns)