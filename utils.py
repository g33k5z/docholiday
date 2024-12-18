import os
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import PromptTemplate


def save_mermaid_graph(
    graph: CompiledStateGraph, filename: str = "diagrams/graph.mermaid"
) -> str:
    """Return the graph in Mermaid (crows feet) format."""
    m = graph.get_graph(xray=True).draw_mermaid()

    with open(filename, "w") as f:
        f.write(m)

    return m


class TemplateRenderer:
    def _path(self, partial: str) -> str:
        return os.path.join(os.path.dirname(__file__), partial)

    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader([self._path("templates/")]),
            autoescape=False,
        )

    async def render_prompt(self, template_name, **kwargs):
        """
        Jinja2 template rendering
        """
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    @staticmethod
    async def get_chat_prompt_message_list(template: str):
        """
        LangChain ChatPromptTemplate, list
        """
        return ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template(template)]
        )

    @staticmethod
    async def get_langchain_prompt(template: str, **kwargs):
        """
        LangChain Prompt, just a string prompt
        """
        prompt = PromptTemplate(
            template=(template), input_variables=[k for k, _ in kwargs]
        )
        return prompt.format(**kwargs)
