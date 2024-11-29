import os
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from genaitor.llama_agent import llama_agent_bp
from redis import Redis
from flasgger import Swagger 

app = Flask(__name__)

csp = {
    'default-src': "'self'"
}

Talisman(app, content_security_policy=csp, force_https=False)

swagger = Swagger(app, template_file='swagger_config.yml')

redis_connection = Redis(host='localhost', port=6379)

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)
limiter.init_app(app)

app.register_blueprint(llama_agent_bp)

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST')
    port = int(os.getenv('FLASK_PORT'))
    debug = os.getenv('FLASK_DEBUG').lower() in ['true', '1']
    print("WARNING: Use Gunicorn or uWSGI in production!")
    
    app.run(host=host, port=port, debug=debug, ssl_context=('cert.pem', 'key.pem'))