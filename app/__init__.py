# app/__init__.py
"""
app package initializer.

This file intentionally keeps package-level imports minimal so the FastAPI
app can import submodules without heavy side-effects (like loading models).
"""

__all__ = [
    "server",
    "model",
    "utils",
    "infer_adapter",
]
