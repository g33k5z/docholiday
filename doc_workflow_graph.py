from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel
from utils import save_mermaid_graph
from doc_loader_graph import DocLoader, doc_loader_graph
from doc_stringer_graph import DocStringer, Update, doc_stringer_graph


llm = ChatOpenAI(model="gpt-4o")


class DocHoliday(BaseModel):
    dir: str = "./"
    glob: str = "**/main.py"
    docs: Optional[List[Document]] = []
    codes: Optional[List[Document]] = []
    updates: Optional[List[Update]] = []
    handled: Optional[List[Update]] = []


async def loader_to_stringer(state: DocLoader) -> DocStringer:
    return DocStringer(
        docs=state.docs, codes=state.codes, dir=state.dir, glob=state.glob
    )


doc_workflow_builder = StateGraph(DocHoliday, input=DocHoliday, output=DocHoliday)


doc_workflow_builder.add_node("doc_loader", doc_loader_graph)
doc_workflow_builder.add_node("loader_to_stringer", loader_to_stringer)
doc_workflow_builder.add_node("doc_stringer", doc_stringer_graph)

doc_workflow_builder.add_edge(START, "doc_loader")
doc_workflow_builder.add_edge("doc_loader", "loader_to_stringer")
doc_workflow_builder.add_edge("loader_to_stringer", "doc_stringer")
doc_workflow_builder.add_edge("doc_stringer", END)
doc_workflow_graph = doc_workflow_builder.compile()
doc_workflow_graph.name = "DocHoliday Workflow"

save_mermaid_graph(doc_workflow_graph, "diagrams/doc_workflow_graph.mermaid")
