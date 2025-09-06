#!/usr/bin/env python3

import base64
import cv2
import numpy as np
from utils import extract_graph_from_image, decode_base64_image, detect_nodes, detect_edges_simple, kruskal_mst

def test_with_local_image():
    """Test the MST calculation with local graph images"""
    
    # Test with graph_0.png
    print("Testing with graph_0.png...")
    
    # Read and encode the image
    with open('graph_0.png', 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Test the main function
    result = extract_graph_from_image(base64_image)
    print(f"MST weight for graph_0.png: {result}")
    print(f"Expected: 53")
    
    # Let's also test step by step for debugging
    print("\nStep-by-step analysis:")
    
    # Decode image
    image = decode_base64_image(base64_image)
    print(f"Image shape: {image.shape}")
    
    # Detect nodes
    nodes = detect_nodes(image)
    print(f"Detected {len(nodes)} nodes: {nodes}")
    
    # Detect edges
    edges = detect_edges_simple(image, nodes)
    print(f"Detected {len(edges)} edges: {edges}")
    
    # Calculate MST
    if edges and nodes:
        mst_weight = kruskal_mst(len(nodes), edges)
        print(f"MST weight: {mst_weight}")
    
    # Test with graph_1.png if it exists
    try:
        print("\n" + "="*50)
        print("Testing with graph_1.png...")
        
        with open('graph_1.png', 'rb') as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        result = extract_graph_from_image(base64_image)
        print(f"MST weight for graph_1.png: {result}")
        print(f"Expected: 77")
        
    except FileNotFoundError:
        print("graph_1.png not found, skipping...")

if __name__ == "__main__":
    test_with_local_image()
