from typing import Any

from pydantic import BaseModel, Field

import planning_agent_demo.callables.base
from planning_agent_demo.ast.result import ResultError, ResultOk


class RunState(BaseModel):
    available_callables: list["planning_agent_demo.callables.base.BaseCallable"] = Field(
        default_factory=lambda: list(planning_agent_demo.callables.base.BaseCallable.__registry__),
        description="The callable functions",
    )

    variables: dict[str, Any] = Field(
        default_factory=dict, description="The current state of the variables"
    )
    result: ResultOk | ResultError | None = None

    @property
    def callables(self):
        mapping = {fn.definition.name: fn for fn in self.available_callables}
        if len(mapping) != len(self.available_callables):
            raise ValueError(
                "Run state provided with multiple functions with the same name, mapping failed"
            )
        return mapping
