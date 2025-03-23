import networkx as nx
import matplotlib.pyplot as plt

# Function to calculate degree centrality manually
def degree_centrality(graph):
    centrality = {}
    total_nodes = graph.number_of_nodes() - 1
    for node in graph.nodes():
        centrality[node] = len(list(graph.neighbors(node))) / total_nodes
    return centrality

# Function to calculate closeness centrality manually
def closeness_centrality(graph):
    centrality = {}
    for node in graph.nodes():
        shortest_paths = nx.single_source_shortest_path_length(graph, node)
        total_distance = sum(shortest_paths.values())
        if total_distance > 0:
            centrality[node] = (len(shortest_paths) - 1) / total_distance
        else:
            centrality[node] = 0
    return centrality

# Function to calculate betweenness centrality manually
def betweenness_centrality(graph):
    centrality = {node: 0 for node in graph.nodes()}
    for source in graph.nodes():
        shortest_paths = nx.single_source_shortest_path_length(graph, source)
        for target, path_length in shortest_paths.items():
            if source != target:
                for node in nx.shortest_path(graph, source=source, target=target):
                    if node != source and node != target:
                        centrality[node] += 1
    # Normalize by dividing by (n-1)(n-2)
    total_nodes = graph.number_of_nodes()
    normalization_factor = (total_nodes - 1) * (total_nodes - 2)
    for node in centrality:
        centrality[node] /= normalization_factor
    return centrality


# Visualization function
def visualize_centrality(graph,top_degree,top_closeness,top_betweenness, filename="static/centrality_graph.png"):
    pos = nx.spring_layout(graph)

    # Assign node colors based on the highest centrality for each node
    node_colors = []
    for node in graph.nodes:
        if node == top_degree[0][0]:
            node_colors.append("red")
            print(f"red {node}")  # Highest Degree Centrality
        elif node == top_closeness[0][0]:
            node_colors.append("orange")
            print(f"orange {node}")  # Highest Closeness Centrality
        elif node == top_betweenness[0][0]:
            node_colors.append("lightgreen")
            print(f"light green {node}")  # Highest Betweenness Centrality
        else:
            node_colors.append("lightblue")  # Default

    # Draw the graph
    plt.figure(figsize=(10, 10))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        font_size=10,
        edge_color="gray",
        alpha=0.8
    )

    # Add a legend for centrality measures
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Degree Centrality', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Closeness Centrality', markerfacecolor='orange', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Betweenness Centrality', markerfacecolor='lightgreen', markersize=10),
           
        ],
        loc="best"
    )

    plt.title("Centrality Visualization")
    plt.savefig(filename)
    plt.close()

