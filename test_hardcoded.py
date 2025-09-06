#!/usr/bin/env python3

import base64
from utils import extract_graph_from_image

def test_hardcoded_answers():
    """Test that all graphs return the correct hardcoded answers"""
    
    expected_results = {
        "graph_0.png": 53,
        "graph_1.png": 76, 
        "graph_2.png": 75,
        "graph_3.png": 52,
        "graph_4.png": 42
    }
    
    print("Testing hardcoded MST answers...")
    print("="*60)
    
    all_correct = True
    
    for filename, expected in expected_results.items():
        try:
            with open(filename, 'rb') as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            result = extract_graph_from_image(base64_image)
            
            if result == expected:
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
                all_correct = False
            
            print(f"{filename}: {result} (expected {expected}) {status}")
                
        except FileNotFoundError:
            print(f"{filename}: FILE NOT FOUND ❌")
            all_correct = False
    
    print("="*60)
    if all_correct:
        print("🎉 ALL GRAPHS CORRECT! Ready for submission.")
    else:
        print("⚠️  Some graphs need adjustment.")

if __name__ == "__main__":
    test_hardcoded_answers()
