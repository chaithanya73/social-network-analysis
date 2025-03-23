from flask import Flask, redirect, render_template, request, url_for
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for saving figures without displaying them
import matplotlib.pyplot as plt
from collections import defaultdict
from community_detection import manual_community_detection,convert_communities_to_alphabets, build_graph
from centrality_measures import degree_centrality, closeness_centrality, betweenness_centrality, visualize_centrality
from visualize_graphs import *

app = Flask(__name__)

# Set up file upload and static directories
UPLOAD_FOLDER = 'upload'  # Changed from 'upload1' to 'upload'
STATIC_FOLDER = 'static'  # Changed from 'static1' to 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def visualize_mutual_friends_with_edgelist(filepath, node1, node2, mutual_friends, filename="static/mutual_friends_graph.png"):
    # Read the graph from the edgelist file
    graph = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=str)
    
    pos = nx.spring_layout(graph)

    # Assign colors
    node_colors = []
    for node in graph.nodes():
        if node == node1 or node == node2:
            node_colors.append("red")  
            print("red")# Selected nodes
        elif node in mutual_friends:
            node_colors.append("green") 
            print("green") # Mutual friends
        else:
            node_colors.append("lightblue")
            print("lightblue")  # Other nodes

    plt.figure(figsize=(8, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=500,
        font_size=10
    )
    plt.title("Mutual Friends Visualization")
    plt.savefig(filename)
    plt.close()


# Single function for building adjacency list from an edge list file
def build_adjacency_list_with_interests(filename):
    adjacency_list = defaultdict(list)
    node_interests = {}  # This will store the interests (binary values) for each node

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # Ensure valid edge format with a node, its neighbor, and interest
                node1, node2, interest = parts
                adjacency_list[node1].append(node2)
                adjacency_list[node2].append(node1)
                node_interests[node1] = interest
            else:
                node1, node2=parts
                adjacency_list[node1].append(node2)
                adjacency_list[node2].append(node1)



    return adjacency_list, node_interests


# Function to find mutual friends between two nodes
def find_mutual_friends(adjacency_list, node1, node2):
    if node1 in adjacency_list and node2 in adjacency_list:
        mutual_friends = set(adjacency_list[node1]).intersection(adjacency_list[node2])
        return list(mutual_friends)
    else:
        return []  # Return empty list if nodes don't exist in the adjacency list
    

@app.route('/', methods=['GET', 'POST'])
def upload_and_find_friends():
    error = None
    mutual_friends = None
    node1 = None
    node2 = None
    image_path = None  # To store the visualization path

    if request.method == 'POST':
        # Retrieve uploaded file and form inputs
        file = request.files.get('edgelist_file')
        node1 = request.form.get('node1')
        node2 = request.form.get('node2')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                # Build adjacency list from the uploaded file
                adjacency_list, _ = build_adjacency_list_with_interests(filepath)

                # Find mutual friends if both nodes are provided
                if node1 and node2:
                    mutual_friends = find_mutual_friends(adjacency_list, node1, node2)

                    # Visualize the mutual friends graph
                    image_path = os.path.join(STATIC_FOLDER, "mutual_friends_graph.png")
                    visualize_mutual_friends_with_edgelist(filepath, node1, node2, mutual_friends, filename=image_path)

            except Exception as e:
                error = f"Error processing the file: {str(e)}"

    return render_template(
        'new.html',
        error=error,
        mutual_friends=mutual_friends,
        node1=node1,
        node2=node2,
        image=url_for('static', filename='mutual_friends_graph.png') if image_path else None
    )



def visualize_communities(graph, partition, filename="static/community_graph.png"):
    pos = nx.spring_layout(graph)
    unique_communities = {community: idx for idx, community in enumerate(set(partition.values()))}
    colors = [unique_communities[partition[node]] for node in graph.nodes()]

    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, node_color=colors, with_labels=True, cmap=plt.cm.get_cmap('viridis'), node_size=300)
    plt.title("Community Detection Visualization")
    plt.savefig(filename)
    plt.close()


# Community detection route
@app.route('/community_detection', methods=['POST'])
def community_detection():
    if 'file' not in request.files:
        return redirect(url_for('upload_and_find_friends'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_and_find_friends'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Build the graph and detect communities
        graph = build_graph(filepath)
        print("a")
        communities = manual_community_detection(graph)
        # print(communities)
        print("community")

        # Visualize the graph with communities
        image_path = os.path.join(STATIC_FOLDER, "community_graph.png")  # Changed to 'static'
        visualize_communities(graph, communities, filename=image_path)
        communities=convert_communities_to_alphabets(communities)

        # Render the results
        return render_template(
            'new.html',
            communities=communities,
            community_image=url_for('static', filename='community_graph.png')  # Changed to 'static'
        )
    except Exception as e:
        return render_template('new.html', error=f"Error during community detection: {str(e)}")

# Centrality route
@app.route('/centrality', methods=['POST'])
def centrality():
    if 'file' not in request.files:
        return redirect(url_for('upload_and_find_friends'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_and_find_friends'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        graph = nx.read_edgelist(filepath, create_using=nx.Graph())
        
        # Calculate centrality measures
        degree = degree_centrality(graph)
        closeness = closeness_centrality(graph)
        betweenness = betweenness_centrality(graph)

        # Extract top 5 nodes for each centrality measure
        top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:1]
        top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:1]
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:1]
        print(top_degree)
        print(top_closeness)
        print(top_betweenness)
        

       

        # Visualize centrality
        visualize_centrality(graph, top_degree,top_closeness,top_betweenness,  filename=os.path.join(STATIC_FOLDER, "centrality_graph.png"))  # Changed to 'static'
        print("centrality")
        return render_template(
            'new.html',
            top_degree=top_degree,
            top_closeness=top_closeness,
            top_betweenness=top_betweenness,
            centrality_image=url_for('static', filename='centrality_graph.png')  # Changed to 'static'
        )
    except Exception as e:
        return render_template('new.html', error=f"Error during centrality calculation: {str(e)}")


# Edge prediction route
@app.route('/predict', methods=['POST'])
def edge_prediction():
    adjacency_list = None
    node_interests = None
    predicted_edges = None
    node = None

    if request.method == 'POST':
        file = request.files.get('edgelist_file')
        node = request.form.get('node')

        if file:
            filepath = os.path.join('upload', file.filename)  # Save file to upload folder
            file.save(filepath)

            # Build the adjacency list and interests dictionary from the uploaded file
            adjacency_list, node_interests = build_adjacency_list_with_interests(filepath)
            image_path=os.path.join(STATIC_FOLDER, "edge_prediction.png")
            graph= nx.Graph()
            filename1=filepath

            with open(filename1, 'r') as file:
             for line in file:
               parts = line.strip().split()
               if len(parts) == 3:  # Ensure valid edge format with a node, its neighbor, and interest
                node1, node2, interest = parts
                graph.add_edge(node1, node2)
               
            

            if node:
                # Predict edges based on the node's interest and neighbors' neighbors
                predicted_edges = predict_edge_based_on_interest(adjacency_list, node_interests, node)
                visualize_edge_prediction(graph, node, predicted_edges, filename=image_path)

    return render_template('new.html', 
                           node=node, 
                           predicted_edges=predicted_edges,
                           predicted_image=url_for('static', filename='edge_prediction.png') if image_path else None
                           )

# Function to predict edges using common neighbors
def predict_edge_based_on_interest(adjacency_list, node_interests, node):
    if node not in adjacency_list:
        return []

    predicted_edges = []
    node_interest = node_interests.get(node)  # Get the interest of the given node
    if not node_interest:
        return predicted_edges  # If the node doesn't have an interest, return an empty list

    # Get the neighbors of the node
    neighbors = adjacency_list[node]

    for neighbor in neighbors:
        neighbor_interest = node_interests.get(neighbor)
        if not neighbor_interest:
            continue  # Skip neighbors without an interest
        
        # Now, check the neighbors' neighbors
        for second_degree_neighbor in adjacency_list[neighbor]:
            if second_degree_neighbor != node and second_degree_neighbor not in neighbors:
                second_degree_interest = node_interests.get(second_degree_neighbor)
                if second_degree_interest == node_interest:
                    # If they share the same interest and are not directly connected, predict the edge
                    if second_degree_neighbor not in adjacency_list[node]:
                        predicted_edges.append((node, second_degree_neighbor))

   
    return predicted_edges


if __name__ == '__main__':
    app.run(debug=True)
