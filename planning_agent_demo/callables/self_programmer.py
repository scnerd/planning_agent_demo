from planning_agent_demo.ast.callable import CallableDefinition, CallableInvocation
from planning_agent_demo.ast.expression import Program
from planning_agent_demo.ast.variable import PlaceholderDefinition
from planning_agent_demo.callables.base import BaseCallable


class SelfProgrammer(BaseCallable):
    name: str
    instructions: str
    inputs: dict[str, PlaceholderDefinition]
    expected_outputs: dict[str, PlaceholderDefinition]
    program: Program | None

    @property
    def definition(self) -> CallableDefinition:
        return CallableDefinition(
            name=self.name,
            description=self.instructions,
            parameters=self.inputs,
            allow_extra_parameters=False,
            returns=self.expected_outputs,
        )

    @property
    def invocation_template(self) -> type[CallableInvocation]:
        arg_model = create_model(

        )


