from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the local CUBIT Inspect inference API."""
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
