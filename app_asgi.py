from app import app
from asgiref.wsgi import WsgiToAsgi

# Create an ASGI application for direct use by ASGI servers
app = WsgiToAsgi(app.wsgi_app) 