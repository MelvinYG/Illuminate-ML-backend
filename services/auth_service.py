"""
Admin authentication service - JWT + bcrypt.
Protects admin-only endpoints like /retrain.
"""
import os
import bcrypt
import jwt
from datetime import datetime, timezone, timedelta
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session
from loguru import logger

from database import get_db
from db_models import AdminUser

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 60 * 12  # 12 hours - admin tool


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET not configured")
    return secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(admin_id: str, email: str) -> str:
    payload = {
        "sub": admin_id,
        "email": email,
        "role": "admin",
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES),
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM])


def seed_admin(db: Session):
    """
    Idempotent admin seed.
    - If admin with ADMIN_EMAIL doesn't exist: create.
    - If exists but the .env password differs: refresh hash.
    """
    email = os.environ.get("ADMIN_EMAIL")
    password = os.environ.get("ADMIN_PASSWORD")
    if not email or not password:
        logger.warning("ADMIN_EMAIL / ADMIN_PASSWORD not set — skipping admin seed")
        return

    existing = db.query(AdminUser).filter(AdminUser.email == email).first()
    if existing is None:
        admin = AdminUser(email=email, password_hash=hash_password(password), name="Admin")
        db.add(admin)
        db.commit()
        logger.info(f"✅ Admin seeded: {email}")
    elif not verify_password(password, existing.password_hash):
        existing.password_hash = hash_password(password)
        db.commit()
        logger.info(f"✅ Admin password refreshed from env: {email}")
    else:
        logger.info(f"Admin already present: {email}")


def get_current_admin(request: Request, db: Session = Depends(get_db)) -> AdminUser:
    """
    FastAPI dependency. Pull JWT from Authorization: Bearer <token>.
    Reject if missing / invalid / expired / not admin.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header[7:].strip()
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access" or payload.get("role") != "admin":
        raise HTTPException(status_code=401, detail="Invalid token type")

    admin = db.query(AdminUser).filter(AdminUser.id == payload["sub"]).first()
    if admin is None:
        raise HTTPException(status_code=401, detail="Admin user not found")
    return admin
