"""
Backward-compatible entry point. The full stack lives in `main.py`.

Run (camera + adaptive controller; no Arduino):
  python traffic_count.py

Same as:
  python main.py --no-serial

Calibrate ROI boxes (North, South, West, East):
  python main.py --calibrate
"""

from main import main

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("--no-serial")
    main()
