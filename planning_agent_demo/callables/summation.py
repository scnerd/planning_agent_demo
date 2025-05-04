from typing import ClassVar

from pydantic import BaseModel, ConfigDict, validate_call

from planning_agent_demo.ast.callable import CallableInvocation, CallableDefinition
from planning_agent_demo.callables.base import BaseCallable, BaseCallableInputs, BaseCallableOutputs, SimpleCallable


class SummationInputs(BaseCallableInputs):
    model_config = ConfigDict(extra="allow")

    a: int
    b: int
    __pydantic_extra__: dict[str, int]


class SummationOutputs(BaseCallableOutputs):
    sum: int


class SummationTool(SimpleCallable[SummationInputs, SummationOutputs]):
    __register_callable__: ClassVar[bool] = True

    name: ClassVar[str] = "summation"
    description: ClassVar[str] = "A tool for summing some numbers"
    inputs: ClassVar[type[BaseCallableInputs]] = SummationInputs
    outputs: ClassVar[type[BaseCallableOutputs]] = SummationOutputs

    @validate_call
    def execute(self, arguments: SummationInputs) -> SummationOutputs:
        return SummationOutputs(sum=sum(
            arguments.model_dump(exclude_unset=True).values()
        ))
