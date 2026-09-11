"""Thin compatibility entry point; prefer main.py."""
from main import main
if __name__=='__main__': raise SystemExit(main())
