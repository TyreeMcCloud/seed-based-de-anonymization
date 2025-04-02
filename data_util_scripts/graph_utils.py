import networkx as nx
import matplotlib.pyplot as plt
import os

def load_graph(edgelist_path):
    """Load a graph from an edgelist file."""
    G = nx.read_edgelist(edgelist_path, nodetype=int)
    return G

def load_seed_mapping(seed_mapping_path):
    """Load seed mappings from a file."""
    seed_mapping = {}
    with open(seed_mapping_path, 'r') as f:
        for line in f:
            g1_node, g2_node = map(int, line.strip().split())
            seed_mapping[g1_node] = g2_node
    return seed_mapping

def visualize_graph(G, title="Graph Visualization", seed_nodes=None, seed_labels=None):
    plt.figure(figsize=(25, 20))  

    # Spread nodes out more to reduce density
    pos = nx.spring_layout(G, k=.2, iterations=150)  
    pos = nx.rescale_layout_dict(pos, scale=10.0)

    # Draw edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', 
            width=1, alpha=0.7, node_size=500, font_size=8)

    if seed_nodes:
        # Draw seed nodes in red
        nx.draw_networkx_nodes(G, pos, nodelist=seed_nodes, node_color='red', node_size=600)

        if seed_labels:
            # Show original node labels (default)
            nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()}, font_color='black', font_size=10)

            # Show seed mapping labels (offset to avoid overlap)
            seed_text_offset = {node: (pos[node][0], pos[node][1] + 0.02) for node in seed_nodes}
            nx.draw_networkx_labels(G, seed_text_offset, labels={node: str(seed_labels[node]) for node in seed_nodes},
                                    font_color='green', font_size=12, font_weight='bold')

    plt.title(title)
    plt.show()


if __name__ == "__main__":
    data_dir = "data/"
    g1_path = os.path.join(data_dir, "seed_G1.edgelist")
    g2_path = os.path.join(data_dir, "seed_G2.edgelist")
    seed_path = os.path.join(data_dir, "seed_mapping.txt")

    G1 = load_graph(g1_path)
    G2 = load_graph(g2_path)
    seed_mapping = load_seed_mapping(seed_path)

    visualize_graph(G1, title="Graph G1", seed_nodes=list(seed_mapping.keys()), seed_labels=seed_mapping)
    reverse_seed_mapping = {v: k for k, v in seed_mapping.items()}
    visualize_graph(G2, title="Graph G2", seed_nodes=list(seed_mapping.values()), seed_labels=reverse_seed_mapping)
