from . import server
import asyncio


def main():
    """Main entry point for the package."""
    server_instance = server.ArxivServer()
    asyncio.run(server_instance.run())


# Optionally expose other important items at package level
__all__ = ["main", "server"]
