from typing import Any

from pydantic import BaseModel, Field

import planning_agent_demo.callables.base
from planning_agent_demo.ast.common import ResultOk, ResultError


class RunState(BaseModel):
    callables: dict[str, "planning_agent_demo.callables.base.BaseCallable"] = Field(default_factory=lambda: {
        callable_instance.definition.name: callable_instance
        for callable_instance in planning_agent_demo.callables.base.BaseCallable.__registry__
    }, description="The callable functions")

    variables: dict[str, Any] = Field(default_factory=dict, description="The current state of the variables")
    result: ResultOk | ResultError | None = None