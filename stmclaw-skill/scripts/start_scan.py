#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

STMCLAW_PATH = r"I:\STMClaw"
SKILL_PATH = Path(__file__).parent.parent


def load_env_file(filepath):
    """Load environment variables from a .env file."""
    if not os.path.exists(filepath):
        return {}
    env_vars = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
    return env_vars


def main():
    parser = argparse.ArgumentParser(description="Start STMClaw scan")
    parser.add_argument(
        '--instruction',
        default="The scan shall start from the top-left and proceed in a serpentine pattern.",
        help='Navigation instruction for the scan',
    )
    args = parser.parse_args()

    # Load the skill configuration file
    config_file = SKILL_PATH / "config.env"
    file_env = load_env_file(config_file)

    # Set environment variables (file values override system environment)
    env = os.environ.copy()
    env.update(file_env)
    env['STMCLAW_NAVIGATION_INSTRUCTION'] = args.instruction

    # Validate required configuration
    if not env.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY not found in config.env or environment")
        sys.exit(1)

    print(f"[STMClaw] Starting scan...")
    print(f"[STMClaw] Config loaded from: {config_file}")

    # Run STMClaw through the adapter
    try:
        subprocess.run(
            [sys.executable, "openclaw_adapter.py"],
            cwd=STMCLAW_PATH,
            env=env,
            check=True,
        )
        print("[STMClaw] Scan completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[STMClaw] Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
