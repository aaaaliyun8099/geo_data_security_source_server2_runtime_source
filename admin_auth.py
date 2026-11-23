from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path
from typing import Any

from .timefmt import row_dict


class AdminAuthStore:
    """SQLite-backed superadmin/admin/user credentials and bearer tokens."""

    def __init__(self, database_path: Path | str, bootstrap_file: Path | str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.bootstrap_file = Path(bootstrap_file)
        self.bootstrap_file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.database_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS admin_users (
                username TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS admin_tokens (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at TEXT
            )"""
        )
        self.db.commit()
        self._bootstrap()

    @staticmethod
    def _hash_secret(value: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt, 240_000)
        return f"pbkdf2_sha256$240000${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_secret(value: str, encoded: str) -> bool:
        try:
            scheme, rounds, salt_hex, digest_hex = encoded.split("$", 3)
            if scheme != "pbkdf2_sha256":
                return False
            actual = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)).hex()
            return hmac.compare_digest(actual, digest_hex)
        except Exception:
            return False

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _bootstrap(self) -> None:
        row = self.db.execute("SELECT username FROM admin_users WHERE role='SUPERADMIN' LIMIT 1").fetchone()
        if row:
            return
        username = "superadmin"
        password = secrets.token_urlsafe(18)
        configured = __import__("os").environ.get("GDS_SUPERADMIN_PASSWORD", "")
        if configured:
            password = configured
        self.db.execute(
            "INSERT INTO admin_users(username,role,password_hash) VALUES(?,?,?)",
            (username, "SUPERADMIN", self._hash_secret(password)),
        )
        self.db.commit()
        if not configured:
            self.bootstrap_file.write_text(
                f"username={username}\npassword={password}\nchange_password_after_first_login=true\n",
                encoding="utf-8",
            )
            try:
                import subprocess
                subprocess.run(["icacls.exe", str(self.bootstrap_file), "/inheritance:r", "/grant:r", "Administrator:F"], check=False, capture_output=True)
            except Exception:
                pass

    def authenticate_password(self, username: str, password: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM admin_users WHERE username=? AND active=1", (username,)).fetchone()
        if not row or not self._verify_secret(password, row["password_hash"]):
            return None
        return {"username": row["username"], "role": row["role"]}

    def issue_token(self, username: str) -> str:
        row = self.db.execute("SELECT username,role,active FROM admin_users WHERE username=?", (username,)).fetchone()
        if not row or not row["active"]:
            raise ValueError("user not found or inactive")
        token = "gdsadm_" + secrets.token_urlsafe(32)
        self.db.execute(
            "INSERT INTO admin_tokens(token_hash,username,role) VALUES(?,?,?)",
            (self._hash_token(token), row["username"], row["role"]),
        )
        self.db.commit()
        return token

    def authenticate_token(self, token: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT t.username,t.role,u.active user_active,t.active token_active FROM admin_tokens t JOIN admin_users u ON u.username=t.username WHERE t.token_hash=?",
            (self._hash_token(token),),
        ).fetchone()
        if not row or not row["user_active"] or not row["token_active"]:
            return None
        self.db.execute("UPDATE admin_tokens SET last_used_at=CURRENT_TIMESTAMP WHERE token_hash=?", (self._hash_token(token),))
        self.db.commit()
        return {"username": row["username"], "role": row["role"]}

    def change_password(self, username: str, old_password: str, new_password: str) -> None:
        row = self.db.execute("SELECT password_hash FROM admin_users WHERE username=? AND active=1", (username,)).fetchone()
        if not row or not self._verify_secret(old_password, row["password_hash"]):
            raise ValueError("old password is invalid")
        self.db.execute("UPDATE admin_users SET password_hash=?,updated_at=CURRENT_TIMESTAMP WHERE username=?", (self._hash_secret(new_password), username))
        self.db.execute("UPDATE admin_tokens SET active=0 WHERE username=?", (username,))
        self.db.commit()

    def create_user(self, username: str, role: str, password: str) -> dict[str, Any]:
        role = role.upper()
        if role not in {"ADMIN", "USER"}:
            raise ValueError("role must be ADMIN or USER")
        self.db.execute("INSERT INTO admin_users(username,role,password_hash) VALUES(?,?,?)", (username, role, self._hash_secret(password)))
        self.db.commit()
        return {"username": username, "role": role, "active": True}

    def list_users(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT username,role,active,created_at,updated_at FROM admin_users ORDER BY username").fetchall()
        return [row_dict(row) for row in rows]

    def set_active(self, username: str, active: bool) -> None:
        self.db.execute("UPDATE admin_users SET active=?,updated_at=CURRENT_TIMESTAMP WHERE username=?", (int(active), username))
        if not active:
            self.db.execute("UPDATE admin_tokens SET active=0 WHERE username=?", (username,))
        self.db.commit()

    def revoke_tokens(self, username: str) -> None:
        self.db.execute("UPDATE admin_tokens SET active=0 WHERE username=?", (username,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
