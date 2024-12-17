from typing import List, Optional, cast
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel
from utils import save_mermaid_graph
import asyncio

llm = ChatOpenAI(model="gpt-4o", max_retries=3)


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
        template=( #TODO: this prompt is grabage
            """
            Given the following code snippet, generate a docstring for the function.
            The function signature is as follows:
            {code}
            Return will be as an Update object with the pattern and replacement fields filled in.
            class Update(BaseModel):
                pattern: str # regex, example: r"(def function_name\\(.*\\):)"
                replacement: str # docstring, example: "\1\n\"\"\"\"Function description.\"\"\""
            

            Only return json of {schema}
            Do not add atitional comment.
            """
        ),
    )

    # Format the input for the LLM
    return prompt_template.format(
        code=doc.page_content, schema=Update.model_json_schema()
    )


async def __generate_docstrings(state: DocStringer) -> DocStringer:
    if not state.codes:
        return state

    if not state.updates:
        state.updates = []

    for code in state.codes:
        prompt = await get_prompt(code)
        update = await llm.with_structured_output(Update).ainvoke(prompt)
        state.updates.append(cast(Update, update))

    return state


async def generate_docstrings(state: DocStringer) -> DocStringer:
    if not state.codes:
        return state

    if not state.updates:
        state.updates = []

    # TODO: retry handling langchain style (https://github.com/langchain-ai/langchain/discussions/24197)
    async def process_code(code):
        prompt = await get_prompt(code)
        update = await llm.with_structured_output(Update).ainvoke(prompt)
        return cast(Update, update)

    # Run all updates in parallel
    tasks = [process_code(code) for code in state.codes]
    results = await asyncio.gather(*tasks)

    state.updates.extend(results)
    return state


async def print_updates(state: DocStringer) -> DocStringer:
    if state.updates:
        for doc in state.updates:
            print(doc)

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
