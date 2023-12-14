from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Load your dataset containing drone patent abstracts
df = pd.read_excel('Dronealexa.xlsx')

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode abstracts
abstract_embeddings = model.encode(df['Abstract'].tolist(), convert_to_tensor=True)

# Simple function to find the most relevant document based on user input
def get_most_similar_document(user_input):
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    user_input_embedding_cpu = user_input_embedding.cpu().numpy()
    similarities = cosine_similarity(user_input_embedding_cpu.reshape(1, -1), abstract_embeddings.cpu().numpy())
    most_similar_index = similarities.argmax()
    max_similarity = similarities[0, most_similar_index]
    threshold = 0.5  
    
    if max_similarity < threshold:
        return "This is out of the given data set."
    
    return df.loc[most_similar_index, 'Abstract']

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')  # Render your HTML file for the interface

# Define route to handle user input
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    response = get_most_similar_document(user_input)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
