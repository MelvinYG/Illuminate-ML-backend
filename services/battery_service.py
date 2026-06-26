import pickle
import numpy as np
from pathlib import Path
from loguru import logger
from exceptions import ModelNotLoadedError

MODEL_PATH = Path("models/battery_model.pkl")
# Load model once when service starts — not on every request

_model_data = None
def load_model_at_startup():
    """
    Call this explicitly at app startup.
    Fails loudly and immediately if model is missing.
    Much better than failing silently on first request.
    """
    global _model_data

    if not MODEL_PATH.exists():
        raise ModelNotLoadedError()

    try:
        with open(MODEL_PATH, "rb") as f:
            _model_data = pickle.load(f)

        # Validate the loaded data has expected keys
        assert "model" in _model_data, "Missing 'model' key in pkl"
        assert "scaler" in _model_data, "Missing 'scaler' key in pkl"

        logger.info(f"✅ Battery model loaded from {MODEL_PATH}")

    except ModelNotLoadedError:
        raise

    except Exception as e:
        raise ModelNotLoadedError() from e