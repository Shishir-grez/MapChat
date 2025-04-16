from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from agent import chatbot, extract_locations, convert_to_coordinates
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_input = data['message']
    print(f"Received message: {user_input}")
    
    raw_response = chatbot.run(user_input).content
    print(f"Raw response from chatbot: {raw_response}")
    
    cleaned_response = re.sub(r'\[LOCATION\](.*?)\[/LOCATION\]', r'\1', raw_response)
    
    locations = extract_locations(raw_response)
    print(f"Extracted locations: {locations}")
    
    coordinates = convert_to_coordinates(locations)
    print(f"Converted coordinates: {coordinates}")
    
    formatted_coordinates = [{"name": loc, "lat": coords[0], "lon": coords[1]} 
                           for loc, coords in coordinates.items() if isinstance(coords, tuple)]
    
    print(f"Formatted coordinates for frontend: {formatted_coordinates}")
    
    response_data = {
        "response": cleaned_response,
        "coordinates": formatted_coordinates
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)