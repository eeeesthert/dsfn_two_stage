"""Thin compatibility entry point. Prefer main.py."""

from main import main

if __name__ == "__main__":
	raise SystemExit(main())
