from typing import Dict, List, Optional, cast
from uuid import UUID
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph.message import uuid
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel, ConfigDict
from utils import save_mermaid_graph
import asyncio

llm = ChatOpenAI(model="gpt-4o", timeout=15, max_retries=3)


class Update(BaseModel):
    id: UUID = uuid.uuid4()
    pattern: str
    replacement: str
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
            Given the following code snippet, generate a docstring for the function.
            The function signature is as follows:
            {code}
            Return will be as an Update object with the pattern and replacement fields filled in.
            Example:
                class Update(BaseModel):
                    pattern: str # regex
                    replacement: str # regex replacement 
            will brecome:
                class Update(BaseModel):
                    \"\"\"
                    Represents a single text update operation using a regex pattern and replacement.

                    Attributes:
                        pattern (str): A regular expression (regex) pattern used to identify the text to match.
                        replacement (str): The string or regex substitution pattern applied to matches of `pattern`.

                    Example:
                        Create an Update instance to replace occurrences of "foo" with "bar":

                        update = Update(pattern=r"\\bfoo\\b", replacement="bar")
                        print(update.pattern)       # Output: "\\bfoo\\b"
                        print(update.replacement)   # Output: "bar"

                    Use Cases:
                        - Define structured text transformations in scripts or pipelines.
                        - Apply multiple regex-based updates programmatically in a workflow.
                    \"\"\"
                    pattern: str # regex
                    replacement: str # regex replacement 

            Only return json of {schema}
            Do not add additional comment.
            """
        ),
    )

    # Format the input for the LLM
    return prompt_template.format(
        code=doc.page_content, schema=Update.model_json_schema()
    )


def is_hashable(obj) -> bool:
    try:
        hash(obj)
        return True
    except TypeError:
        return False


async def generate_docstrings(state: DocStringer) -> DocStringer:
    if not state.codes:
        return state

    if not state.updates:
        state.updates = {}

    # TODO: retry handling langchain style (https://github.com/langchain-ai/langchain/discussions/24197)
    async def process_code(code):
        prompt = await get_prompt(code)
        update = await llm.with_structured_output(Update).ainvoke(prompt)
        build_up = cast(Update, update)
        build_up.doc = code
        return build_up

    # Run all updates in parallel
    tasks = [process_code(code) for code in state.codes]
    results = await asyncio.gather(*tasks)

    # state.updates.update(results)
    for update in results:
        try:
            state.updates[update.id] = update
        except Exception as e:
            print(e)
    return state


async def print_updates(state: DocStringer) -> DocStringer:
    if state.updates:
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
doc_stringer_builder.add_node("print_updates", print_updates)

doc_stringer_builder.add_edge(START, "generate_docstrings")
doc_stringer_builder.add_edge("generate_docstrings", "print_updates")
doc_stringer_builder.add_edge("print_updates", END)
doc_stringer_graph = doc_stringer_builder.compile()
doc_stringer_graph.name = "Source Code DocStringer"

save_mermaid_graph(doc_stringer_graph, "diagrams/doc_stringer_graph.mermaid")
