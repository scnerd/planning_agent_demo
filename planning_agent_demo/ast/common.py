from typing import Any, ForwardRef, Literal

from pydantic import BaseModel, Field

import planning_agent_demo.callables.base


class ResultOk(BaseModel):
    result_type: Literal["ok"] = Field("ok", frozen=True)
    values: dict[str, Any] = Field(..., description="The values returned by the function")


class ResultError(BaseModel):
    result_type: Literal["error"] = Field("error", frozen=True)
    error: str = Field(..., description="The error message")


class RunState(BaseModel):
    callables: dict[str, "planning_agent_demo.callables.base.BaseCallable"] = Field(default_factory=lambda: {
        callable_instance.definition.name: callable_instance
        for callable_instance in planning_agent_demo.callables.base.BaseCallable.__registry__
    }, description="The callable functions")

    variables: dict[str, Any] = Field(default_factory=dict, description="The current state of the variables")
    result: ResultOk | ResultError | None = None
