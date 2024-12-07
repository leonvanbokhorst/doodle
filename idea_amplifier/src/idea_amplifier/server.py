from typing import Any, Sequence, Optional
import asyncio
import json
from datetime import datetime
import os
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
import networkx as nx
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server
import numpy as np
import matplotlib.pyplot as plt
import io
import base64



class IdeaGraphDB:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def init_constraints(self):
        """Initialize Neo4j constraints"""
        async with self.driver.session() as session:
            # Ensure unique IDs for ideas
            await session.run(
                "CREATE CONSTRAINT idea_id IF NOT EXISTS "
                "FOR (i:Idea) REQUIRE i.id IS UNIQUE"
            )

    async def add_idea(self, content: str, idea_type: str) -> int:
        """Add a new idea to the graph"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MERGE (id:UniqueId{name: 'Idea'})
                ON CREATE SET id.count = 1
                ON MATCH SET id.count = id.count + 1
                WITH id.count as uniqueId
                CREATE (i:Idea {
                    id: uniqueId,
                    content: $content,
                    type: $type,
                    timestamp: datetime()
                })
                RETURN i.id as id
                """,
                content=content,
                type=idea_type,
            )
            record = await result.single()
            return record["id"]

    async def connect_ideas(
        self, from_id: int, to_id: int, relationship: str, strength: float = 0.5
    ):
        """Create a connection between two ideas"""
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (i1:Idea {id: $from_id})
                MATCH (i2:Idea {id: $to_id})
                CREATE (i1)-[r:RELATES {
                    type: $rel_type,
                    strength: $strength,
                    timestamp: datetime()
                }]->(i2)
                """,
                from_id=from_id,
                to_id=to_id,
                rel_type=relationship,
                strength=strength,
            )

    async def get_recent_ideas(self, limit: int = 5) -> list[dict]:
        """Get the most recent ideas"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (i:Idea)
                RETURN i.id as id, i.content as content, 
                       i.type as type, i.timestamp as timestamp
                ORDER BY i.timestamp DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(record) async for record in result]

    async def get_idea_neighborhood(self, idea_id: int, depth: int = 2) -> dict:
        """Get an idea and its related ideas up to certain depth"""
        async with self.driver.session() as session:
            # Build query with proper depth range
            query = f"""
                MATCH path = (center:Idea {{id: $id}})-[r:RELATES*0..{depth}]-(connected:Idea)
                WITH collect(DISTINCT center) + collect(DISTINCT connected) as nodes,
                     collect(DISTINCT relationships(path)) as rels
                UNWIND rels as rel_list
                UNWIND rel_list as r
                WITH nodes, collect(DISTINCT {{
                    from: startNode(r).id,
                    to: endNode(r).id,
                    type: r.type,
                    strength: r.strength
                }}) as relationships
                RETURN 
                    [n IN nodes | {{
                        id: n.id,
                        content: n.content,
                        type: n.type
                    }}] as nodes,
                    relationships
                """
            result = await session.run(query, id=idea_id)
            return dict(await result.single())

    async def find_idea_clusters(self, min_cluster_size: int = 2) -> list[dict]:
        """Find clusters of related ideas using Neo4j's GDS library"""
        async with self.driver.session() as session:
            # First, create a graph projection
            await session.run(
                """
                CALL gds.graph.project.cypher(
                    'idea-clusters',
                    'MATCH (i:Idea) RETURN id(i) AS id, i.content AS content',
                    'MATCH (i1:Idea)-[r:RELATES]->(i2:Idea) 
                     RETURN id(i1) AS source, id(i2) AS target, r.strength AS weight'
                )
                """
            )

            # Run community detection
            result = await session.run(
                """
                CALL gds.louvain.stream('idea-clusters')
                YIELD nodeId, communityId
                WITH communityId, collect(nodeId) AS nodeIds
                WHERE size(nodeIds) >= $minSize
                MATCH (i:Idea)
                WHERE id(i) IN nodeIds
                RETURN communityId,
                       collect({id: i.id, content: i.content}) as ideas
                ORDER BY communityId
                """,
                minSize=min_cluster_size,
            )

            clusters = [dict(record) async for record in result]

            # Cleanup projection
            await session.run("CALL gds.graph.drop('idea-clusters')")

            return clusters


class IdeaAmplifierServer:
    def __init__(self):
        self.server = Server("idea-amplifier")
        self.db = IdeaGraphDB(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="testing123",  # Change in production!
        )
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="ideagraph://recent",
                    name="Recent Ideas",
                    description="Your most recent ideas",
                ),
                Resource(
                    uri="ideagraph://clusters",
                    name="Idea Clusters",
                    description="Clusters of related ideas",
                ),
                Resource(
                    uri="ideagraph://neighborhood/{id}",
                    name="Idea Neighborhood",
                    description="An idea and its connections",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri == "ideagraph://recent":
                ideas = await self.db.get_recent_ideas()
                return json.dumps(ideas, indent=2)

            elif uri == "ideagraph://clusters":
                clusters = await self.db.find_idea_clusters()
                return json.dumps(clusters, indent=2)

            elif uri.startswith("ideagraph://neighborhood/"):
                idea_id = int(uri.split("/")[-1])
                neighborhood = await self.db.get_idea_neighborhood(idea_id)
                return json.dumps(neighborhood, indent=2)

            raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="capture_idea",
                    description="Save a new idea",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The idea content",
                            },
                            "type": {
                                "type": "string",
                                "enum": [
                                    "thought",
                                    "question",
                                    "concept",
                                    "connection",
                                ],
                                "description": "Type of idea",
                            },
                        },
                        "required": ["content", "type"],
                    },
                ),
                Tool(
                    name="connect_ideas",
                    description="Connect two ideas",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "from_id": {"type": "integer"},
                            "to_id": {"type": "integer"},
                            "relationship": {"type": "string"},
                            "strength": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["from_id", "to_id", "relationship"],
                    },
                ),
                Tool(
                    name="visualize_neighborhood",
                    description="Create a visual graph of an idea's neighborhood",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "idea_id": {"type": "integer"},
                            "depth": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 3,
                                "default": 2,
                            },
                        },
                        "required": ["idea_id"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Any
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            try:
                if name == "capture_idea":
                    idea_id = await self.db.add_idea(
                        arguments["content"], arguments["type"]
                    )
                    return [
                        TextContent(type="text", text=f"Idea saved with ID: {idea_id}")
                    ]

                elif name == "connect_ideas":
                    await self.db.connect_ideas(
                        arguments["from_id"],
                        arguments["to_id"],
                        arguments["relationship"],
                        arguments.get("strength", 0.5),
                    )
                    return [
                        TextContent(
                            type="text", text="Connection created successfully!"
                        )
                    ]

                elif name == "visualize_neighborhood":
                    neighborhood = await self.db.get_idea_neighborhood(
                        arguments["idea_id"], arguments.get("depth", 2)
                    )

                    G = nx.Graph()
                    for node in neighborhood["nodes"]:
                        G.add_node(node["id"], content=node["content"], type=node["type"])
                    for rel in neighborhood["relationships"]:
                        G.add_edge(rel["from"], rel["to"], type=rel["type"], strength=rel["strength"])

                    plt.figure(figsize=(10, 10))
                    pos = nx.spring_layout(G)
                    nx.draw(G, pos,
                        with_labels=True,
                        node_color="lightblue",
                        node_size=1000,
                        font_size=8,
                        labels={n: G.nodes[n]["content"][:20] + "..." for n in G.nodes()}
                    )

                    # Save to PNG
                    png_io = io.BytesIO()
                    plt.savefig(png_io, format='png')
                    plt.close()
                    
                    # Convert binary data to base64 string
                    png_base64 = base64.b64encode(png_io.getvalue()).decode('utf-8')
                    
                    return [ImageContent(
                        type="image",
                        mimeType="image/png",
                        data=png_base64
                    )]

                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Neo4jError as e:
                return [
                    TextContent(
                        type="text", text=f"Database error: {str(e)}", isError=True
                    )
                ]

    async def run(self):
        # Initialize database constraints
        await self.db.init_constraints()

        # Start MCP server
        async with stdio_server() as transport:
            try:
                await self.server.run(
                    transport[0],
                    transport[1],
                    self.server.create_initialization_options(),
                )
            finally:
                await self.db.close()


async def main():
    server = IdeaAmplifierServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
