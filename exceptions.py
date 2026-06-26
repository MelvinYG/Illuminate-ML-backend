"""
Custom exception hierarchy for Illuminate.
Think of these like custom Error classes in JavaScript.
"""

class IlluminateError(Exception):
    """Base exception for all Illuminate errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class WeatherAPIError(IlluminateError):
    """Weather API is down or returned bad data."""
    def __init__(self, message: str):
        super().__init__(message, status_code=503)  # 503 = Service Unavailable


class ModelNotLoadedError(IlluminateError):
    """ML model .pkl file missing or corrupted."""
    def __init__(self):
        super().__init__(
            "ML model not loaded. Run train_model.py first.",
            status_code=503
        )


class OptimizationFailedError(IlluminateError):
    """PuLP optimizer returned infeasible or error status."""
    def __init__(self, status: str):
        super().__init__(
            f"Optimization failed with status: {status}",
            status_code=422  # 422 = Unprocessable Entity
        )


class UserNotFoundError(IlluminateError):
    """User ID doesn't exist in DB."""
    def __init__(self, user_id: str):
        super().__init__(
            f"User {user_id} not found",
            status_code=404
        )


class DatabaseError(IlluminateError):
    """DB query or connection failed."""
    def __init__(self, message: str):
        super().__init__(f"Database error: {message}", status_code=503)


class InvalidModeError(IlluminateError):
    """Invalid optimization mode passed."""
    VALID_MODES = ["self_consumption", "tou_savings", "full_backup", "low_power"]

    def __init__(self, mode: str):
        super().__init__(
            f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}",
            status_code=422
        )