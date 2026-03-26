"""Common helper functions used across the project."""

import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_model(model, path: str):
    """Serialize model to disk. Scikit-learn or Keras."""
    if hasattr(model, 'save'):
        # keras
        model.save(path)
    else:
        import joblib
        joblib.dump(model, path)


def load_model(path: str):
    if path.endswith('.h5') or path.endswith('.keras'):
        from tensorflow.keras.models import load_model
        return load_model(path)
    else:
        import joblib
        return joblib.load(path)
