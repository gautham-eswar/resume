from app import app
from asgiref.wsgi import WsgiToAsgi

# Create an ASGI application by wrapping the Flask WSGI app
asgi_app = WsgiToAsgi(app.wsgi_app) 