from flask import Flask, request, jsonify
import secrets
import string
import sqlite3

app = Flask(__name__)

DATABASE = 'genaitor.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                username TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                permissions TEXT,
                UNIQUE(username)
            )
        ''')
    conn.commit()
    conn.close()

def generate_api_key():
    alphabet = string.ascii_letters + string.digits
    api_key = ''.join(secrets.choice(alphabet) for _ in range(32))
    return api_key

def save_api_key(username, api_key):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
            INSERT INTO api_keys (username, api_key)
            VALUES (?, ?)
            ON CONFLICT(username) DO UPDATE SET 
                api_key = excluded.api_key,
                created_at = CURRENT_TIMESTAMP
        ''', (username, api_key))    
    conn.commit()
    conn.close()

def get_api_key(username):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT api_key FROM api_keys WHERE username=?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

@app.route('/generate-api-key', methods=['POST'])
def generate_api_key_route():
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    new_api_key = generate_api_key()
    save_api_key(username, new_api_key)

    return jsonify({"api_key": new_api_key})

@app.route('/get-api-key/<username>', methods=['GET'])
def get_api_key_route(username):
    api_key = get_api_key(username)
    
    if not api_key:
        return jsonify({"error": "API key not found for the user"}), 404
    
    return jsonify({"api_key": api_key})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)
