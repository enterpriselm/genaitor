from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from config import config
from transformers import AutoTokenizer, AutoModel
import torch

paper_similarity_bp = Blueprint('paper_similarity', __name__)

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')

# In-memory storage for paper embeddings (for demonstration)
papers_db = []

def embed_text(text):
    """
    Function to generate embeddings for a given text using Hugging Face embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()

@paper_similarity_bp.route('/add-paper', methods=['POST'])
def add_paper():
    """
    Endpoint to add a paper review with its text.
    Expects JSON data with 'title' and 'review'.
    """
    data = request.get_json()
    title = data.get('title')
    review = data.get('review')

    if not title or not review:
        return jsonify({"error": "Both title and review are required"}), 400

    # Embed the review text and store it
    embedding = embed_text(review)
    papers_db.append({"title": title, "review": review, "embedding": embedding})

    return jsonify({"message": "Paper review added successfully"}), 200

@paper_similarity_bp.route('/find-similar', methods=['POST'])
def find_similar():
    """
    Endpoint to find the most similar paper review to the user’s request.
    Expects JSON data with 'query'.
    """
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "A query is required"}), 400

    # Embed the user query
    query_embedding = embed_text(query)

    # Compute similarities with each stored paper embedding
    similarities = [
        {
            "title": paper["title"],
            "similarity": cosine_similarity(query_embedding, paper["embedding"]).flatten()[0]
        }
        for paper in papers_db
    ]

    # Sort papers by highest similarity score
    most_similar_paper = max(similarities, key=lambda x: x["similarity"])

    return jsonify({
        "most_similar_paper": most_similar_paper["title"],
        "similarity_score": most_similar_paper["similarity"]
    })
