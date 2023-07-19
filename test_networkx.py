import networkx as nx

if __name__ == "__main__":
    G = nx.complete_graph(5)
    print(G.out_edges(G.nodes[0], data=True))