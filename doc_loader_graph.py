import re
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel
from utils import save_mermaid_graph



class DocLoader(BaseModel):
    dir: str = "./"
    glob: str = "**/main.py"
    docs: Optional[List[Document]] = []
    codes: Optional[List[Document]] = []


class CustomPythonTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter for Python with custom split patterns."""

    custom_python_patterns = [
        r"\nclass ",  # Split at class definitions
        r"\nasync def ",  # Split at function definitions
        r"\ndef ",  # Split at function definitions
        r"\n\n\n[a-zA-Z_]+_builder",
        # r"\n\n",        # Split at newlines
        # r" ",         # Split at spaces
    ]

    def __init__(self, chunk_size=1000, chunk_overlap=0):
        # Define custom split patterns for Python
        super().__init__(
            separators=self.custom_python_patterns,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def hand_rolled_splitter(self, text: str) -> List[str]:
        """Custom splitter for Python text."""
        current_start = 0
        splits = []
        for pattern in self.custom_python_patterns:
            for match in re.finditer(pattern, text):
                end = match.start()
                chunk = text[current_start:end].strip()
                if chunk:
                    splits.append(chunk)
                current_start = end
        if current_start < len(text):
            splits.append(text[current_start:])
        return splits


async def load_docs(state: DocLoader) -> DocLoader:

    loader = DirectoryLoader(
        state.dir, glob=state.glob, loader_cls=TextLoader, show_progress=False
    )
    # loader = TextLoader(__file__)
    docs = loader.load()

    python_splitter = CustomPythonTextSplitter(  # RecursiveCharacterTextSplitter(
        # language=Language.PYTHON,
        # separators=custom_python_patterns,
        chunk_size=100,
        chunk_overlap=0,
    )
    # python_docs = python_splitter.create_documents(
    #     [x.page_content for x in docs], [x.metadata for x in docs]
    # )
    python_docs = []

    for x in docs:
        for split in python_splitter.hand_rolled_splitter(x.page_content):
            python_docs.append(Document(page_content=split, metadata=x.metadata))

    return DocLoader(
        docs=python_docs, codes=python_docs, dir=state.dir, glob=state.glob
    )


async def print_docs(state: DocLoader) -> DocLoader:
    if state.docs:
        for doc in state.docs:
            print(doc.metadata)

    if state.codes:
        for code in state.codes:
            print(code.metadata)
            print(code.page_content)
            print("\n---------------\n\n")

    return state


doc_loader_builder = StateGraph(DocLoader, input=DocLoader, output=DocLoader)


doc_loader_builder.add_node("load_docs", load_docs)
doc_loader_builder.add_node("print_docs", print_docs)

doc_loader_builder.add_edge(START, "load_docs")
doc_loader_builder.add_edge("load_docs", "print_docs")
doc_loader_builder.add_edge("print_docs", END)
doc_loader_graph = doc_loader_builder.compile()
doc_loader_graph.name = "Source Code Loader"

save_mermaid_graph(doc_loader_graph, "diagrams/doc_loader_graph.mermaid")
