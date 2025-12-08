class graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, u, v):
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)  # For undirected graph

    def remove_edge(self, u, v):
        if u in self.adjacency_list and v in self.adjacency_list[u]:
            self.adjacency_list[u].remove(v)
        if v in self.adjacency_list and u in self.adjacency_list[v]:
            self.adjacency_list[v].remove(u)

    def get_neighbors(self, u):
        return self.adjacency_list.get(u, [])

    def __str__(self):
        return str(self.adjacency_list)
    

# Example usage:
if __name__ == "__main__":    
    g = graph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    print("Graph adjacency list:", g)
    print("Neighbors of 1:", g.get_neighbors(1))
    g.remove_edge(1, 2)
    print("Graph after removing edge (1, 2):", g)