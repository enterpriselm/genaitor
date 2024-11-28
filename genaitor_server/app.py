import os
from flask import Flask
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from genaitor.llama_agent import llama_agent_bp

app = Flask(__name__)

Talisman(app)

limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

app.register_blueprint(llama_agent_bp)

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST')
    port = int(os.getenv('FLASK_PORT'))
    debug = os.getenv('FLASK_DEBUG').lower() in ['true', '1']
    print("WARNING: Use Gunicorn or uWSGI in production!")
    
    app.run(host=host, port=port, debug=debug)