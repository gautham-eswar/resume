from app import app
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio

# Create an ASGI application
asgi_app = app 