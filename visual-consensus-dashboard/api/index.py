"""Vercel serverless entry point for Dash app."""

import sys
import os

# Add parent directory to path so src.* imports work
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)

# Change working directory so relative paths (data/, assets/) resolve correctly
os.chdir(parent_dir)

from app import app as dash_app

# Vercel Python runtime looks for a WSGI callable named 'app'
app = dash_app.server
