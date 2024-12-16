import re
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel
from utils import save_mermaid_graph

llm = ChatOpenAI(model="gpt-4o")


class Update(BaseModel):
    pattern: str
    replacement: str


class DocStringer(BaseModel):
    dir: str = "./"
    glob: str = "**/main.py"
    docs: Optional[List[Document]] = []
    codes: Optional[List[Document]] = []
    updates: Optional[List[Update]] = []
    handled: Optional[List[Update]] = []


async def get_prompt(doc: Document) -> str:
    # Create a prompt for the LLM
    prompt_template = PromptTemplate(
        input_variables=["code", "schema"],
        template=(
            """
            Given the following code snippet, generate a docstring for the function.
            The function signature is as follows:
            {code}
            Return will be as an Update object with the pattern and replacement fields filled in.
            class Update(BaseModel):
                pattern: str
                replacement: str

            Only return json of {schema}
            Do not add atitional comment.
            """
        ),
    )

    # Format the input for the LLM
    return prompt_template.format(code=doc.page_content, schema=Update.model_json_schema())




async def generate_docstrings(state: DocStringer) -> DocStringer:
    for code in state.codes:
        docstring = await llm.generate_text(code.page_content)
        docstring = re.sub(r"\n+", "\n", docstring)
        docstring = re.sub(r"\n", "\n\n", docstring)
        docstring = re.sub(r"\n\n\n+", "\n\n", docstring)
        docstring = docstring.strip()
        docstring = docstring.replace("\n", "\n# ")
        docstring = f"# {docstring}"


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
