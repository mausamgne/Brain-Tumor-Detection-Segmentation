import os
import sqlite3
import hashlib
import secrets
import string
from datetime import datetime, timedelta

# Consistent database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

def init_db():
    """Initialize the database with required tables"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_verified BOOLEAN DEFAULT FALSE
            )
        ''')
        # Password reset tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        # NEW: Add segmentation reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS segmentation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                tumor_area REAL DEFAULT 0,
                perimeter REAL DEFAULT 0,
                circularity REAL DEFAULT 0,
                eccentricity REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                class_distribution TEXT,
                volume_data TEXT,  -- NEW: Store volume data as JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Optional: enforce case-insensitive unique email
        cursor.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower
            ON users (LOWER(email))
        ''')
        conn.commit()

def hash_password(password: str) -> str:
    """Hash a password using SHA256 + salt"""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a password against the stored salted hash"""
    try:
        salt, hashed = stored_password.split('$')
        return hashed == hashlib.sha256((salt + provided_password).encode()).hexdigest()
    except Exception:
        return False

def check_password_strength(password: str):
    """Check if a password is strong and return suggestions"""
    suggestions = []
    strong = True
    if len(password) < 8:
        strong = False
        suggestions.append("At least 8 characters long")
    if not any(c.islower() for c in password):
        strong = False
        suggestions.append("Include lowercase letters")
    if not any(c.isupper() for c in password):
        strong = False
        suggestions.append("Include uppercase letters")
    if not any(c.isdigit() for c in password):
        strong = False
        suggestions.append("Include numbers")
    if not any(c in string.punctuation for c in password):
        strong = False
        suggestions.append("Include special characters")
    return strong, suggestions

def generate_strong_password(length: int = 12) -> str:
    """Generate a strong random password"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def init_reports_table():
    """Initialize the reports table for storing PDF reports"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                tumor_volume REAL DEFAULT 0,
                tumor_area REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                growth_stage TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()

# Call this function in your existing init_db() function or after it
init_reports_table()

# Run init_db when module is imported
init_db()
