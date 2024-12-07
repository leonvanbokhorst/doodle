from setuptools import setup, find_packages

setup(
    name="idea-amplifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "neo4j",
        "networkx",
        "numpy",
        "mcp-core",  # Assuming this is the correct package name for MCP
    ],
    python_requires=">=3.7",
) 