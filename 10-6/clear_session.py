#!/usr/bin/env python3
"""
Clear Saved Session

This utility script safely clears the saved FMLS session cookies.
Use this when you want to force a fresh login or if cookies are corrupted.

Usage:
    python clear_session.py
"""

import sys
from pathlib import Path
from config import Settings


def main():
    """Clear saved session cookies."""
    print("\n" + "=" * 80)
    print("Clear Saved FMLS Session".center(80))
    print("=" * 80 + "\n")

    cookie_file = Settings.COOKIES_DIR / "fmls_session.json"

    print(f"Cookie file: {cookie_file}")

    if not cookie_file.exists():
        print("\n✓ No session file found - nothing to clear.")
        print("  (Session is already clean)")
        return 0

    # Show file info before deletion
    stat = cookie_file.stat()
    file_size = stat.st_size

    print(f"\nFound session file:")
    print(f"  Size: {file_size:,} bytes")
    print(f"  Location: {cookie_file}")

    # Confirm deletion
    print("\nThis will delete the saved session and you'll need to login with 2FA again.")
    response = input("Continue? (y/N): ").strip().lower()

    if response != 'y':
        print("\nCancelled - session file not deleted.")
        return 0

    # Delete the file
    try:
        cookie_file.unlink()
        print("\n✓ Session file deleted successfully!")
        print("\nNext time you run main.py:")
        print("  1. You'll be prompted for 2FA login")
        print("  2. New session will be saved after successful authentication")
        print("  3. Subsequent runs will use the new session (no 2FA)")
        return 0

    except Exception as e:
        print(f"\n✗ ERROR deleting session file: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)
