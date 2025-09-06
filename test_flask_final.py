#!/usr/bin/env python3

import base64
import json
from app import app

def test_flask_with_all_graphs():
    """Test the Flask endpoint with all 5 graphs"""
    
    print("Testing Flask endpoint with all graphs...")
    
    # Read all test graphs
    test_cases = []
    graph_files = ["graph_0.png", "graph_1.png", "graph_2.png", "graph_3.png", "graph_4.png"]
    expected_results = [53, 76, 75, 52, 42]
    
    for filename in graph_files:
        try:
            with open(filename, 'rb') as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                test_cases.append({"image": base64_image})
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    
    # Test using Flask test client
    with app.test_client() as client:
        response = client.post('/mst-calculation', 
                             data=json.dumps(test_cases),
                             content_type='application/json')
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"Results: {results}")
            
            print("\nDetailed comparison:")
            for i, (expected, result) in enumerate(zip(expected_results[:len(results)], results)):
                actual = result['value']
                status = "✅" if actual == expected else "❌"
                print(f"Graph {i}: Expected {expected}, Got {actual} {status}")
        else:
            print(f"Error: {response.get_json()}")

if __name__ == "__main__":
    test_flask_with_all_graphs()
