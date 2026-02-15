"""CLI entry point for Autobot -- Living Entity."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="autobot",
        description="Autobot -- A Living Entity",
    )
    sub = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = sub.add_parser("serve", help="Start the Autobot server.")
    serve_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000).",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0).",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    serve_parser.add_argument(
        "--identity",
        type=str,
        default=None,
        help="Path to identity config file (YAML or JSON).",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print("Error: uvicorn is required. Install with: pip install uvicorn[standard]")
            sys.exit(1)

        import os
        if args.identity:
            os.environ["AUTOBOT_IDENTITY"] = args.identity

        print(f"Starting Autobot on http://{args.host}:{args.port}")
        uvicorn.run(
            "autobot.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
