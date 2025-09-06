from flask import Flask, request, jsonify, make_response, send_file
import os
# import base64
# import cv2
# import numpy as np
# from utils import extract_graph_from_image


app = Flask(__name__)

@app.route("/")
def root():
    resp = make_response(jsonify({"service": "mst-calculator", "status": "ok"}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/chasetheflag", methods=["POST"])
def chasetheflag():
    return jsonify({
    "challenge1": "your_flag_1",
    "challenge2": "",
    "challenge3": "",
    "challenge4": "",
    "challenge5": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
    })

@app.route("/payload_crackme", methods=["GET"])
def payload_crackme():
    file_path = os.path.join("./", "payload_crackme")
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)

@app.route("/payload_sqlinject", methods=["GET"])
def payload_sqlinject():
    file_path = os.path.join("./", "payload_sqlinject")
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)

# @app.route("/mst-calculation", methods=["POST"])
# def mst_calculation():
#     try:
#         # Get the request data
#         data = request.get_json()
        
#         if not data or not isinstance(data, list):
#             return make_response(jsonify({"error": "Invalid input format. Expected a list of test cases."}), 400)
        
#         results = []
        
#         for test_case in data:
#             if not isinstance(test_case, dict) or "image" not in test_case:
#                 return make_response(jsonify({"error": "Each test case must have an 'image' field."}), 400)
            
#             base64_image = test_case["image"]
            
#             # Process the image and calculate MST
#             mst_weight = extract_graph_from_image(base64_image)
            
#             results.append({"value": mst_weight})
        
#         # Return the results with proper headers
#         resp = make_response(jsonify(results), 200)
#         resp.headers["Content-Type"] = "application/json"
#         return resp
        
#     except Exception as e:
#         return make_response(jsonify({"error": f"Internal server error: {str(e)}"}), 500)


if __name__ == "__main__":
    # For local development only
    app.run()
