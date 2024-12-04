import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, List, Sequence
import arxiv
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    CallToolResult,
    ImageContent,
    EmbeddedResource,
)
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arxiv-server")

# Cache configuration
CACHE_DURATION = timedelta(hours=1)
cached_results = {}


class ArxivServer:
    def __init__(self):
        # Server just wants a name - that's it!
        self.server = Server(name="arxiv-server")

        self.client = arxiv.Client()
        self.setup_handlers()

    async def fetch_papers(self, query: str, max_results: int = 10) -> List[dict]:
        cache_key = f"{query}:{max_results}"
        now = datetime.now()

        # Check cache first
        if cache_key in cached_results:
            timestamp, results = cached_results[cache_key]
            if now - timestamp < CACHE_DURATION:
                logger.info("Returning cached results for %s", query)
                return results

        # If not in cache or expired, fetch new results
        logger.info("Fetching new results from arXiv for %s", query)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        try:
            # Get results synchronously since arxiv.Client.results() returns an iterator
            results = []
            for paper in self.client.results(search):
                results.append(
                    {
                        "title": paper.title,
                        "summary": paper.summary,
                        "authors": [author.name for author in paper.authors],
                        "published": paper.published.isoformat(),
                        "url": paper.pdf_url,
                        "categories": paper.categories,
                    }
                )
                if len(results) >= max_results:
                    break

            # Update cache
            cached_results[cache_key] = (now, results)
            return results

        except Exception as e:
            logger.error(f"Error fetching papers: {str(e)}")
            raise

    def setup_handlers(self):
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="arxiv://recent/ai",
                    name="Recent AI Papers",
                    description="Latest artificial intelligence papers from arXiv",
                    mimeType="application/json",
                ),
                Resource(
                    uri="arxiv://recent/cs",
                    name="Recent CS Papers",
                    description="Latest computer science papers from arXiv",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: AnyUrl) -> str:
            uri_str = str(uri)
            if not uri_str.startswith("arxiv://recent/"):
                raise ValueError(f"Unknown resource: {uri}")

            category = uri_str.split("/")[-1]
            query = f"cat:{category}"
            papers = await self.fetch_papers(query)
            return json.dumps(papers, indent=2)

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="search_papers",
                    description="Search for papers on arXiv",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query or keywords",
                            },
                            "max_results": {
                                "type": "number",
                                "description": "Maximum number of results (1-50)",
                                "minimum": 1,
                                "maximum": 50,
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                )
            ]

        @self.server.call_tool()  # noqa
        async def call_tool(
            name: str, arguments: Any
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            if name != "search_papers":
                raise ValueError(f"Unknown tool: {name}")

            query = arguments["query"]
            max_results = int(arguments.get("max_results", 10))

            try:
                papers = await self.fetch_papers(query, max_results)
                # Return just the sequence of content directly
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(papers, indent=2),
                    )
                ]
            except Exception as e:
                return [
                    TextContent(
                        type="text",
                        text=f"Error searching papers: {str(e)}",
                    )
                ]

    async def run(self):
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


# No need for main() here as it's in __init__.py
