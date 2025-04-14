"""
WSGI entry point for the Flask application.
This file is used by gunicorn to run the application.
"""

from app import app

if __name__ == "__main__":
    app.run() 