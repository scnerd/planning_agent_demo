from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from planning_agent_demo.ast.dtype import BaseDtype


class VariableDefinition(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    dtype: BaseDtype


class RuntimeVariable(BaseModel):
    meta: VariableDefinition
    value: Any


class PlaceholderDefinition(VariableDefinition):
    description: str
