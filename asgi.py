from app import app
from asgiref.wsgi import WsgiToAsgi

# Wrap the Flask app in an ASGI wrapper
asgi_app = WsgiToAsgi(app) 