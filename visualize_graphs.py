import networkx as nx
import matplotlib.pyplot as plt

def visualize_edge_prediction(graph, node, predicted_edges, filename="static/edge_prediction.png"):

    pos = nx.spring_layout(graph)
    
    # Determine colors for nodes
    node_colors = []
    for n in graph.nodes:
        if n == node:
            node_colors.append("red")  # Selected node
        elif n in graph.neighbors(node):
            node_colors.append("orange")  # Neighbors
        elif n in predicted_edges:
            node_colors.append("lightblue")  # Other nodes
        else:
            node_colors.append("lightblue")   
    
    # Add predicted edges to the graph for visualization
    temp_graph = graph.copy()
    temp_graph.add_edges_from(predicted_edges)
    
    edge_colors = []
    edge_style=[]
    for edge in temp_graph.edges:
        if edge in predicted_edges or (edge[1], edge[0]) in predicted_edges:
            edge_colors.append("lightgreen")  # Predicted edges
        else:
            edge_colors.append("gray")  # Existing edges
    
    # Plot the graph
    plt.figure(figsize=(10, 10))
    nx.draw(
        temp_graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=500,
        font_size=10
    )
    plt.title(f"Edge Prediction for Node {node}")
    plt.savefig(filename)
    plt.close()
