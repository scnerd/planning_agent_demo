from typing import Literal, Any

from pydantic import BaseModel, Field

from planning_agent_demo.ast.base import BaseExpression
from planning_agent_demo.ast.variable import PlaceholderDefinition


# class ParameterDefinition(PlaceholderDefinition):
#     pass
#
#
# class ReturnDefinition(PlaceholderDefinition):
#     pass


class CallableDefinition(BaseModel):
    name: str
    description: str

    parameters: dict[str, PlaceholderDefinition]
    allow_extra_parameters: bool

    returns: dict[str, PlaceholderDefinition]


class CallableInvocation(BaseExpression):
    expr_type: Literal["func_call"] = Field("func_call", frozen=True)

    name: str
    arguments: dict[str, BaseExpression]

    def __str__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.arguments.items())})"

    def evaluate(self, run_state: "import planning_agent_demo.ast.run_state.RunState") -> Any:
        callable_instance = run_state.callables[self.name]
        args = {key: value.evaluate(run_state) for key, value in self.arguments.items()}
        args = callable_instance.inputs_type.model_validate(args)
        result = callable_instance.execute(args)
        return result.model_dump()


import planning_agent_demo.ast.run_state
