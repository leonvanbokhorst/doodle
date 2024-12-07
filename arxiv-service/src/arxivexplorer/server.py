import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, List, Sequence, Optional, Dict
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
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arxiv-server")

# Cache configuration
CACHE_DURATION = timedelta(hours=1)
cached_results = {}
pending_approvals: Dict[str, dict] = {}


class ArxivServer:
    def __init__(self):
        self.server = Server(name="arxiv-server")
        # Configure client with no caching
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3,  # Ensure we don't hit rate limits
            num_retries=3
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(persist_directory="./data"))
        self.collection = self.chroma_client.get_or_create_collection(name="arxiv_papers")
        
        self.setup_handlers()

    async def store_paper(self, paper_data: dict) -> bool:
        try:
            # Generate paper ID
            paper_id = f"paper_{hash(paper_data['url'])}"
            
            # Check if paper already exists
            try:
                existing = self.collection.get(
                    ids=[paper_id],
                    include=['metadatas']
                )
                if existing and existing['ids']:
                    logger.info(f"Paper already exists in vector store: {paper_data['title']}")
                    return True
            except Exception as e:
                logger.debug(f"Error checking for existing paper: {str(e)}")
                # Continue with storage attempt if check fails
                
            # Store the paper in the vector store
            self.collection.add(
                documents=[paper_data["summary"]],
                metadatas=[{
                    "title": paper_data["title"],
                    "url": paper_data["url"],
                    "authors": ", ".join(paper_data["authors"]),
                    "published": paper_data["published"],
                    "categories": ", ".join(paper_data["categories"])
                }],
                ids=[paper_id]
            )
            logger.info(f"Successfully stored new paper: {paper_data['title']}")
            return True
        except Exception as e:
            logger.error(f"Error storing paper: {str(e)}")
            return False

    async def request_paper_storage(self, paper_data: dict) -> str:
        """Request approval to store a paper. Returns approval token."""
        approval_token = f"approval_{hash(paper_data['url'])}"
        pending_approvals[approval_token] = paper_data
        return approval_token

    async def fetch_papers(self, query: str, max_results: int = 10) -> List[dict]:
        logger.info("Creating arXiv search with parameters:")
        logger.info("  Query: %r", query)
        logger.info("  Max results: %d", max_results)
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            results = []
            async_results = self.client.results(search)
            logger.info("Got results from arXiv API, processing...")
            
            for paper in async_results:
                logger.debug("Processing paper: %s", paper.title)
                results.append({
                    "title": paper.title,
                    "summary": paper.summary,
                    "authors": [author.name for author in paper.authors],
                    "published": paper.published.isoformat(),
                    "url": paper.pdf_url,
                    "categories": paper.categories,
                })
                if len(results) >= max_results:
                    break

            logger.info("Processed %d papers", len(results))
            return results

        except Exception as e:
            logger.error(f"Error fetching papers: {str(e)}", exc_info=True)
            raise

    async def fetch_paper_by_id(self, paper_id: str) -> Optional[dict]:
        """Fetch a specific paper by its arxiv ID."""
        logger.info(f"Fetching paper by ID: {paper_id}")
        try:
            search = arxiv.Search(id_list=[paper_id])
            results = list(self.client.results(search))
            if results:
                paper = results[0]
                return {
                    "title": paper.title,
                    "summary": paper.summary,
                    "authors": [author.name for author in paper.authors],
                    "published": paper.published.isoformat(),
                    "url": paper.pdf_url,
                    "categories": paper.categories,
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching paper by ID: {str(e)}")
            return None

    def setup_handlers(self):
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            # Return empty list since we're using tools instead
            return []

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
                ),
                Tool(
                    name="request_paper_storage",
                    description="Request approval to store a paper in the vector database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_url": {
                                "type": "string",
                                "description": "The URL of the paper to store",
                            }
                        },
                        "required": ["paper_url"],
                    },
                ),
                Tool(
                    name="approve_paper_storage",
                    description="Approve storing a paper in the vector database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "approval_token": {
                                "type": "string",
                                "description": "The approval token for the paper storage request",
                            }
                        },
                        "required": ["approval_token"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Any
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            if name == "search_papers":
                query = arguments["query"]
                max_results = int(arguments.get("max_results", 10))

                try:
                    papers = await self.fetch_papers(query, max_results)
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
            
            elif name == "request_paper_storage":
                paper_url = arguments["paper_url"]
                # Extract arxiv ID from URL
                paper_id = paper_url.split('/')[-1].replace('.pdf', '')
                
                # Try to fetch the paper directly
                paper_data = await self.fetch_paper_by_id(paper_id)
                
                if paper_data:
                    approval_token = await self.request_paper_storage(paper_data)
                    return [
                        TextContent(
                            type="text",
                            text=f"Storage request created. Please approve using the token: {approval_token}",
                        )
                    ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text="Could not fetch paper details. Please verify the URL is correct.",
                        )
                    ]

            elif name == "approve_paper_storage":
                approval_token = arguments["approval_token"]
                if approval_token not in pending_approvals:
                    return [
                        TextContent(
                            type="text",
                            text="Invalid or expired approval token.",
                        )
                    ]

                paper_data = pending_approvals.pop(approval_token)
                success = await self.store_paper(paper_data)
                return [
                    TextContent(
                        type="text",
                        text="Paper successfully stored in the vector database" if success
                        else "Failed to store paper in the vector database",
                    )
                ]
            
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self):
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )
