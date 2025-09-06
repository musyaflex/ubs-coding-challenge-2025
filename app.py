from flask import Flask, request, jsonify, make_response

try:
    # When running as a package: `python -m ubs_server.app` or `flask run` with APP=ubs_server.app
    from .utils import roman_to_int, parse_english_number, parse_german_number, chinese_to_int, classify_representation
except Exception:
    # When running directly: `python app.py`
    from utils import roman_to_int, parse_english_number, parse_german_number, chinese_to_int, classify_representation
from collections import defaultdict, deque, Counter
import re, math

app = Flask(__name__)