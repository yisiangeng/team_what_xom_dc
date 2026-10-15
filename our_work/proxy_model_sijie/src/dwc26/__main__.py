"""Entry point: python -m dwc26 <command> ..."""

import sys

from our_work.proxy_model_sijie.src.dwc26.cli import main

if __name__ == "__main__":
    sys.exit(main())
