import asyncio
import logging
from dotenv import load_dotenv

from doc_loader_graph import doc_loader_graph
from doc_workflow_graph import doc_workflow_graph

load_dotenv()

logging.basicConfig(level=logging.INFO)


async def run_doc_loader_graph(config: dict = {"dir": "./", "glob": "**/main.py"}):
    await doc_loader_graph.ainvoke(config)


async def doc_holiday(config: dict = {"dir": "./", "glob": "**/*.py"}):
    await doc_workflow_graph.ainvoke(config)


if __name__ == "__main__":
    # asyncio.run(run_doc_loader_graph({"dir": "./", "glob": "**/*graph.py"}))
    asyncio.run(doc_holiday({"dir": "./", "glob": "**/*.py"}))
