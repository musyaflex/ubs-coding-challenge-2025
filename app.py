from flask import Flask, request, jsonify, make_response


app = Flask(__name__)

@app.route("/")
def root():
    resp = make_response(jsonify({"service": "ticketing-agent", "status": "ok"}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


if __name__ == "__main__":
    # For local development only
    app.run()
