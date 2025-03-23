import networkx as nx

def calculate_modularity(graph, partition, degrees, m):
    modularity = 0
    for node in graph.nodes():
        community = partition[node]
        k_i = degrees[node]
        k_in = sum(1 for neighbor in graph.neighbors(node) if partition[neighbor] == community)
        modularity += (k_in / m) - (k_i / (2 * m))**2
    return modularity

# Function to calculate modularity
def calculate_modularity_gain(graph, node, community, partition, degrees, m):
    k_i = degrees[node]  # Degree of the node
    k_in = sum(1 for neighbor in graph.neighbors(node) if partition[neighbor] == community)  # Internal connections
    k_tot = sum(degrees[n] for n in graph.nodes if partition[n] == community)  # Total degree of the community
    
    modularity_gain = (k_in / m) - ((k_i * k_tot) / (2 * m)**2)
    return modularity_gain



# Louvain-like community detection
def manual_community_detection(graph, threshold=1e-6, max_iterations=100):
    degrees = dict(graph.degree())
    m = sum(degrees.values()) / 2  # Total edge weight
    partition = {node: node for node in graph.nodes()}  # Initialize each node to its own community
    current_modularity = calculate_modularity(graph, partition, degrees, m)
    improvement = True
    iteration_count = 0

    while improvement and iteration_count < max_iterations:
        improvement = False
        iteration_count += 1

        for node in graph.nodes():
            current_community = partition[node]
            best_community = current_community
            best_gain = 0

            for neighbor in graph.neighbors(node):
                neighbor_community = partition[neighbor]
                if neighbor_community != current_community:
                    gain = calculate_modularity_gain(graph, node, neighbor_community, partition, degrees, m)
                    if gain > best_gain:
                        best_gain = gain
                        best_community = neighbor_community

            if best_gain > 0:
                partition[node] = best_community
                improvement = True

        # Recalculate modularity after each iteration
        new_modularity = calculate_modularity(graph, partition, degrees, m)
        print(f"Iteration {iteration_count}: Modularity = {new_modularity}")
        
        if abs(new_modularity - current_modularity) < threshold:
            print("Modularity improvement below threshold. Stopping.")
            break

        current_modularity = new_modularity

    if iteration_count >= max_iterations:
        print("Maximum iterations reached. Stopping.")

    return partition

def convert_communities_to_alphabets(community_dict):
    import string

    # Generate alphabet labels (single and extended for large numbers of communities)
    alphabets = list(string.ascii_uppercase)  # Single letters: A-Z
    extended_alphabets = [a + b for a in alphabets for b in alphabets]  # Double letters: AA, AB, ...
    labels = alphabets + extended_alphabets  # Combine single and double letters

    # Map unique community numbers to alphabet labels
    unique_communities = sorted(set(community_dict.values()))
    community_mapping = {community: labels[idx] for idx, community in enumerate(unique_communities)}

    # Create a new dictionary with converted community labels
    converted_dict = {node: community_mapping[community] for node, community in community_dict.items()}

    return converted_dict



# converted_dict = convert_communities_to_alphabets(community_dict)
# print(converted_dict)


# Build the graph from an edge list
def build_graph(file_path):
    graph = nx.read_edgelist(file_path, create_using=nx.Graph())
    return graph
