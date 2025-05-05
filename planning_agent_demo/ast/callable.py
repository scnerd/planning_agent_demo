from pydantic import BaseModel

from planning_agent_demo.ast.variable import PlaceholderDefinition


class CallableDefinition(BaseModel):
    name: str
    description: str

    parameters: dict[str, PlaceholderDefinition]
    allow_extra_parameters: bool

    returns: dict[str, PlaceholderDefinition]
