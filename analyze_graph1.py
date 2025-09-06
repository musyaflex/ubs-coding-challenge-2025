#!/usr/bin/env python3

import base64
from utils import decode_base64_image, detect_nodes, detect_edges_simple

def analyze_graph1():
    """Analyze graph_1.png structure"""
    
    print("Analyzing graph_1.png...")
    
    # Read graph_1.png
    with open('graph_1.png', 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Decode image
    image = decode_base64_image(base64_image)
    
    # Detect nodes
    nodes = detect_nodes(image)
    print(f"Detected {len(nodes)} nodes:")
    for i, (x, y) in enumerate(nodes):
        print(f"  Node {i}: ({x}, {y})")
    
    # Detect edges
    edges = detect_edges_simple(image, nodes)
    print(f"\nDetected {len(edges)} edges:")
    for start, end, weight in edges:
        print(f"  Edge: Node {start} <-> Node {end}, weight: {weight}")
    
    # Calculate current total
    total_weight = sum(edge[2] for edge in edges)
    print(f"\nTotal edge weights: {total_weight}")
    
    # Show what MST would select
    from utils import kruskal_mst
    mst_weight = kruskal_mst(len(nodes), edges)
    print(f"Current MST weight: {mst_weight}")
    print(f"Target MST weight: 77")
    print(f"Difference needed: {77 - mst_weight}")

if __name__ == "__main__":
    analyze_graph1()
