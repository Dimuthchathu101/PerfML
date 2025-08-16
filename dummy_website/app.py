from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import time
import random
import threading
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulate database operations
class MockDatabase:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
    
    def get_data(self, key):
        with self.lock:
            time.sleep(random.uniform(0.01, 0.1))  # Simulate DB latency
            return self.data.get(key, {"message": "Data not found"})
    
    def set_data(self, key, value):
        with self.lock:
            time.sleep(random.uniform(0.05, 0.2))  # Simulate DB write latency
            self.data[key] = value
            return {"message": "Data stored successfully"}

db = MockDatabase()

@app.route('/')
def home():
    """Homepage with heavy content to test frontend performance"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/data/<key>')
def get_data(key):
    """Get data endpoint with variable latency"""
    # Simulate different response times based on key
    if key.startswith('heavy'):
        time.sleep(random.uniform(0.5, 2.0))  # Heavy operation
    elif key.startswith('medium'):
        time.sleep(random.uniform(0.1, 0.5))  # Medium operation
    else:
        time.sleep(random.uniform(0.01, 0.1))  # Light operation
    
    result = db.get_data(key)
    return jsonify(result)

@app.route('/api/data', methods=['POST'])
def create_data():
    """Create data endpoint"""
    data = request.get_json()
    key = data.get('key', 'default')
    value = data.get('value', {})
    
    result = db.set_data(key, value)
    return jsonify(result)

@app.route('/api/search')
def search():
    """Search endpoint with complex processing"""
    query = request.args.get('q', '')
    
    # Simulate search processing time
    processing_time = len(query) * 0.01 + random.uniform(0.1, 0.5)
    time.sleep(processing_time)
    
    # Simulate search results
    results = [
        {"id": i, "title": f"Result {i}", "content": f"Content for {query} - {i}"}
        for i in range(1, min(11, len(query) + 5))
    ]
    
    return jsonify({
        "query": query,
        "results": results,
        "processing_time": processing_time
    })

@app.route('/api/users')
def get_users():
    """Get users endpoint with pagination"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    
    # Simulate pagination processing
    time.sleep(random.uniform(0.05, 0.3))
    
    users = [
        {
            "id": i,
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "created_at": datetime.now().isoformat()
        }
        for i in range((page - 1) * limit + 1, page * limit + 1)
    ]
    
    return jsonify({
        "users": users,
        "page": page,
        "limit": limit,
        "total": 1000
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """File upload endpoint"""
    # Simulate file processing
    time.sleep(random.uniform(0.5, 3.0))
    
    return jsonify({
        "message": "File uploaded successfully",
        "filename": "uploaded_file.txt",
        "size": random.randint(1000, 10000)
    })

@app.route('/api/error/<error_type>')
def simulate_error(error_type):
    """Endpoint to simulate different types of errors"""
    if error_type == '500':
        return jsonify({"error": "Internal Server Error"}), 500
    elif error_type == '404':
        return jsonify({"error": "Not Found"}), 404
    elif error_type == '400':
        return jsonify({"error": "Bad Request"}), 400
    elif error_type == 'timeout':
        time.sleep(30)  # Simulate timeout
        return jsonify({"message": "This should timeout"})
    else:
        return jsonify({"error": "Unknown error type"}), 400

@app.route('/api/slow')
def slow_endpoint():
    """Endpoint with intentionally slow response"""
    time.sleep(random.uniform(2, 5))
    return jsonify({"message": "Slow response completed"})

@app.route('/api/memory')
def memory_intensive():
    """Memory-intensive operation"""
    # Simulate memory usage
    large_list = [random.random() for _ in range(100000)]
    result = sum(large_list)
    return jsonify({"result": result, "message": "Memory-intensive operation completed"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
