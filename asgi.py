from app import app
from asgiref.wsgi import WsgiToAsgi

# Create an ASGI application
asgi_app = WsgiToAsgi(app.wsgi_app) 