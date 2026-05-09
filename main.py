from __future__ import annotations

import sys
from subprocess import run


def main() -> None:
    run(
        [sys.executable, "-m", "chainlit", "run", "chainlit_app.py", *sys.argv[1:]],
        check=False,
    )


if __name__ == "__main__":
    main()
