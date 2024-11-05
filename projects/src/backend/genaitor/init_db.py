import sqlite3

def initialize_db():
    conn = sqlite3.connect('genaitor.db')
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            api_key TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            permissions TEXT)
        """)
    
    cursor.execute("INSERT OR IGNORE INTO api_keys (username, api_key) VALUES (?, ?)", 
                   ("test_user", "fc288116-2bd1-442d-abe2-a8cc6e5e0111"))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    initialize_db()
