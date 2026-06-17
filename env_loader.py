"""
Load environment variables from a local .env file.

On Render, Heroku, etc., secrets are injected into the process environment; no .env file
is present and none is required if DATABASE_URL is already set.
"""

import os
from pathlib import Path


def load_env():
    """Merge project `.env` into os.environ if the file exists.

    If there is no `.env` file, only warn when DATABASE_URL is missing (typical local dev).
    Hosted environments set DATABASE_URL in the dashboard; no warning in that case.
    """
    if os.environ.get("MONSTRA_PREVIEW_SERVICE"):
        return

    env_file = Path(__file__).parent / ".env"

    if not env_file.exists():
        if os.environ.get("DATABASE_URL"):
            return
        print(f"Warning: .env file not found at {env_file}")
        print("Create one by copying .env.example: cp .env.example .env")
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Set environment variable
                os.environ[key] = value


if __name__ == "__main__":
    load_env()
    print(f"DATABASE_URL: {os.environ.get('DATABASE_URL', 'NOT SET')}")
