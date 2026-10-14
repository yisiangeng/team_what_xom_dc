"""Entry point: python -m dwc26 <command> ..."""

import sys

from dwc26.cli import main

if __name__ == "__main__":
    sys.exit(main())
