#!/usr/bin/env python3
"""Prompt in the remote terminal; credentials never enter shell command arguments."""
import getpass
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ.get("H3_ROOT", "/data/minimax-h3"))
    directory = root / "private"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    config = directory / "ngrok.yml"
    policy = directory / "ngrok-policy.json"
    if config.exists() or policy.exists():
        raise SystemExit("Private configuration already exists; edit it locally to change credentials.")
    token = getpass.getpass("ngrok agent authtoken: ").strip()
    username = input("Browser username [h3]: ").strip() or "h3"
    if not username.isascii() or not username or ":" in username or "${" in username:
        raise SystemExit("Use an ASCII username without ':' or '${'.")
    password = getpass.getpass("Browser password (20-128 ASCII characters): ")
    if not token or not 20 <= len(password) <= 128 or not password.isascii() or "${" in password:
        raise SystemExit("Invalid token/password; no files written.")
    if password != getpass.getpass("Repeat browser password: "):
        raise SystemExit("Passwords do not match; no files written.")
    values = [(config, {"version": "3", "agent": {"authtoken": token}}),
              (policy, {"on_http_request": [{"actions": [{"type": "basic-auth", "config": {
                  "realm": "MiniMax H3", "enforce": True, "credentials": [username + ":" + password]}}]}]})]
    for path, value in values:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
    print("Saved owner-only ngrok configuration and enforced login policy. No secrets printed.")


if __name__ == "__main__":
    main()
