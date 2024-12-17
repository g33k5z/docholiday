from typing import Dict, List, Optional
from uuid import UUID, uuid4
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from utils import save_mermaid_graph
import asyncio
import os
from aiofiles import open as aio_open  # Asynchronous file handling

llm = ChatOpenAI(model="gpt-4o-mini", timeout=15, max_retries=3)


class Update(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    pattern: str = Field(..., title="Pattern to find in the code")
    replacement: str = Field(..., title="Replacement for the pattern")
    doc: Optional[Document] = None


class DocStringer(BaseModel):
    dir: str = "./"
    glob: str = "**/main.py"
    docs: Optional[List[Document]] = []
    codes: Optional[List[Document]] = []
    updates: Optional[Dict[UUID, Update]] = {}
    handled: Optional[Dict[UUID, Update]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


async def get_prompt(doc: Document) -> str:
    # Create a prompt for the LLM
    prompt_template = PromptTemplate(
        input_variables=["code", "schema"],
        template=(  # TODO: this prompt is grabage
"""
Update the class or def code with a docstring in the Google style. If only imports are present, add a module docstring. For other misc code, add a comment. 

<code>{code}</code>


Maintain the existing code structure char for char, only with added docstring.
Do not wrap code in backticks such as ```python
Do not add additional comment.
"""
        ),
    )

    # Format the input for the LLM
    return prompt_template.format(
        code=doc.page_content, schema=Update.model_json_schema()
    )


async def generate_docstrings(state: DocStringer) -> DocStringer:
    if not state.codes:
        return state

    if not state.updates:
        state.updates = {}

    # TODO: retry handling langchain style (https://github.com/langchain-ai/langchain/discussions/24197)
    async def process_code(code: Document):
        prompt = await get_prompt(code)
        update = await llm.ainvoke(prompt)
        build_up = Update(pattern=code.page_content, replacement=str(update.content))
        build_up.doc = code
        return build_up

    tasks = [process_code(code) for code in state.codes]
    results = await asyncio.gather(*tasks)

    for update in results:
        try:
            state.updates[update.id] = update
        except Exception as e:
            print(e)
    return state


async def apply_updates(state: DocStringer) -> DocStringer:
    if not state.docs or not state.updates:
        return state

    async def apply_single_update(doc: Document, updates: List[Update]) -> Document:
        content = doc.page_content
        for update in updates:
            try:
                buffer = "\n"
                if any(keyword in update.pattern for keyword in ["class ", "def "]):
                    buffer = "\n\n\n"
                content = content.replace(update.pattern, buffer + update.replacement)
            except Exception as e:
                print(f"Failed to apply update {update.id}: {e}")
        return Document(page_content=content, metadata=doc.metadata)

    updates_by_doc = {}
    for update in state.updates.values():
        if update.doc and "uuid" in update.doc.metadata:
            updates_by_doc.setdefault(update.doc.metadata["uuid"], []).append(update)

    tasks = [
        apply_single_update(doc, updates_by_doc.get(doc.metadata["uuid"], []))
        for doc in state.docs
    ]
    updated_docs = await asyncio.gather(*tasks)

    state.docs = updated_docs
    return state


async def write_updated_docs(state: DocStringer) -> DocStringer:
    if not state.docs:
        return state

    async def write_single_doc(doc: Document) -> None:
        if not doc.metadata or "source" not in doc.metadata:
            raise ValueError(
                "Document metadata must contain 'source' indicating the file path."
            )

        original_path = doc.metadata["source"]
        directory, filename = os.path.split(original_path)
        base_name, extension = os.path.splitext(filename)

        updated_filename = f"output/{base_name}_updated{extension}"
        updated_path = os.path.join(directory, updated_filename)

        async with aio_open(updated_path, "w", encoding="utf-8") as f:
            await f.write(doc.page_content)

        print(f"Document saved: {updated_path}")

    write_tasks = [write_single_doc(doc) for doc in state.docs]
    await asyncio.gather(*write_tasks)

    return state


async def print_updates(state: DocStringer) -> DocStringer:
    if state.updates:
        print(f"Generated {len(state.updates)} docstrings")
        for id, update in state.updates.items():
            print(id)
            print(update.doc.metadata if update.doc else "No metadata")
            print(update.pattern)
            print("=>")
            print(update.replacement)
            print("\n------------------------------\n")

    return state


doc_stringer_builder = StateGraph(DocStringer, input=DocStringer, output=DocStringer)


doc_stringer_builder.add_node("generate_docstrings", generate_docstrings)
doc_stringer_builder.add_node("apply_updates", apply_updates)
doc_stringer_builder.add_node("write_updated_docs", write_updated_docs)
doc_stringer_builder.add_node("print_updates", print_updates)

doc_stringer_builder.add_edge(START, "generate_docstrings")
doc_stringer_builder.add_edge("generate_docstrings", "apply_updates")
doc_stringer_builder.add_edge("apply_updates", "write_updated_docs")
doc_stringer_builder.add_edge("write_updated_docs", "print_updates")
doc_stringer_builder.add_edge("print_updates", END)
doc_stringer_graph = doc_stringer_builder.compile()
doc_stringer_graph.name = "Source Code DocStringer"

save_mermaid_graph(doc_stringer_graph, "diagrams/doc_stringer_graph.mermaid")
