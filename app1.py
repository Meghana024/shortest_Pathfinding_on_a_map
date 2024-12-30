from flask import Flask, request, render_template, jsonify
import osmnx as ox
import networkx as nx
import heapq
from datetime import datetime
import polyline

app = Flask(__name__)

place_name = "Adilabad, Telangana, India"
G = ox.graph_from_place(place_name, network_type='drive')

def calculate_route_stats(path, G):
    total_distance = 0
    total_time = 0  
    turn_points = []
    
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i+1])[0]
        total_distance += edge_data.get('length', 0)
        speed = edge_data.get('speed_kph', 30)
        total_time += (edge_data.get('length', 0) / 1000) / (speed / 3600)

        if i > 0:
            prev_bearing = ox.bearing.calculate_bearing(
                G.nodes[path[i-1]]['y'], G.nodes[path[i-1]]['x'],
                G.nodes[path[i]]['y'], G.nodes[path[i]]['x']
            )
            next_bearing = ox.bearing.calculate_bearing(
                G.nodes[path[i]]['y'], G.nodes[path[i]]['x'],
                G.nodes[path[i+1]]['y'], G.nodes[path[i+1]]['x']
            )
            if abs(prev_bearing - next_bearing) > 30: 
                turn_points.append({
                    'lat': G.nodes[path[i]]['y'],
                    'lon': G.nodes[path[i]]['x']
                })
    
    return {
        'distance': round(total_distance / 1000, 2),
        'estimated_time': round(total_time / 60, 1),
        'turn_points': turn_points
    }

def heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def bidirectional_a_star(graph, start, goal):
    if start == goal:
        return [start]
        
    forward_open = []
    backward_open = []
    
    heapq.heappush(forward_open, (0, start))
    heapq.heappush(backward_open, (0, goal))

    forward_came_from = {}
    backward_came_from = {}
    
    forward_g_score = {node: float('inf') for node in graph.nodes}
    backward_g_score = {node: float('inf') for node in graph.nodes}
    
    forward_g_score[start] = 0
    backward_g_score[goal] = 0
    
    start_coords = (graph.nodes[start]['y'], graph.nodes[start]['x'])
    goal_coords = (graph.nodes[goal]['y'], graph.nodes[goal]['x'])
    
    best_path_length = float('inf')
    meeting_point = None
    
    while forward_open and backward_open:
        current_f = heapq.heappop(forward_open)[1]
        
        for neighbor, data in graph[current_f].items():
            edge_length = data[0].get('length', None)
            if edge_length is None:
                current_coords = (graph.nodes[current_f]['y'], graph.nodes[current_f]['x'])
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])
                edge_length = ((current_coords[0] - neighbor_coords[0]) ** 2 + 
                             (current_coords[1] - neighbor_coords[1]) ** 2) ** 0.5
                
            tentative_g_score = forward_g_score[current_f] + edge_length
            
            if tentative_g_score < forward_g_score[neighbor]:
                forward_came_from[neighbor] = current_f
                forward_g_score[neighbor] = tentative_g_score
                
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])
                f_score = tentative_g_score + heuristic(neighbor_coords, goal_coords)
                
                heapq.heappush(forward_open, (f_score, neighbor))
                
                if backward_g_score[neighbor] != float('inf'):
                    path_length = forward_g_score[neighbor] + backward_g_score[neighbor]
                    if path_length < best_path_length:
                        best_path_length = path_length
                        meeting_point = neighbor
        
        current_b = heapq.heappop(backward_open)[1]
        
        for neighbor, data in graph[current_b].items():
            edge_length = data[0].get('length', None)
            if edge_length is None:
                current_coords = (graph.nodes[current_b]['y'], graph.nodes[current_b]['x'])
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])
                edge_length = ((current_coords[0] - neighbor_coords[0]) ** 2 + 
                             (current_coords[1] - neighbor_coords[1]) ** 2) ** 0.5
                
            tentative_g_score = backward_g_score[current_b] + edge_length
            
            if tentative_g_score < backward_g_score[neighbor]:
                backward_came_from[neighbor] = current_b
                backward_g_score[neighbor] = tentative_g_score
                
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])
                f_score = tentative_g_score + heuristic(neighbor_coords, start_coords)
                
                heapq.heappush(backward_open, (f_score, neighbor))
                
                if forward_g_score[neighbor] != float('inf'):
                    path_length = forward_g_score[neighbor] + backward_g_score[neighbor]
                    if path_length < best_path_length:
                        best_path_length = path_length
                        meeting_point = neighbor
    
    if meeting_point is None:
        return None

    forward_path = reconstruct_path(forward_came_from, meeting_point)
    backward_path = reconstruct_path(backward_came_from, meeting_point)
    
    if forward_path is None or backward_path is None:
        return None
    
    backward_path.pop()
    backward_path.reverse()
    return forward_path + backward_path

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/find_path', methods=['POST'])
def find_path():
    data = request.get_json()
    start_coords = (float(data['start_lat']), float(data['start_lon']))
    end_coords = (float(data['end_lat']), float(data['end_lon']))

    start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    shortest_path = bidirectional_a_star(G, start_node, end_node)
    
    if shortest_path:
        path_coords = [
            {'lat': G.nodes[node]['y'], 'lon': G.nodes[node]['x']} 
            for node in shortest_path
        ]
        
        stats = calculate_route_stats(shortest_path, G)
    
        path_tuples = [(coord['lat'], coord['lon']) for coord in path_coords]
        encoded_path = polyline.encode(path_tuples)
        
        return jsonify({
            'path': path_coords,
            'encoded_path': encoded_path,
            'distance': stats['distance'],
            'estimated_time': stats['estimated_time'],
            'turn_points': stats['turn_points'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        return jsonify({
            'error': 'No path found',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

if __name__ == '__main__':
    app.run(debug=True)
