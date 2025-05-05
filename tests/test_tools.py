import pytest
from pydantic import ConfigDict, ValidationError

from planning_agent_demo.ast.expression import (
    AssignmentStatement,
    LiteralExpr,
    Program,
    ReturnStatement,
    VariableExpr,
    CallableInvocation,
)
from planning_agent_demo.ast.result import ResultOk
from planning_agent_demo.ast.run_state import RunState
from planning_agent_demo.callables.base import BaseCallableInputs
from planning_agent_demo.callables.summation import SummationInputs, SummationOutputs, SummationTool


def test_summation_directly():
    summation_tool = SummationTool()
    assert summation_tool.execute(SummationInputs(a=1, b=2, c=4)) == SummationOutputs(sum=7)


def test_summation_as_invocation():
    run_state = RunState(
        variables=dict(
            x=1,
            y=2,
            z=4,
        )
    )
    invocation = CallableInvocation(
        name="summation",
        arguments=dict(
            a=VariableExpr(name="x"),
            b=VariableExpr(name="y"),
            c=VariableExpr(name="z"),
        ),
    )
    result = invocation.evaluate(run_state)
    assert result == dict(sum=7)


def test_summation_in_program():
    run_state = RunState(
        variables=dict(
            x=1,
            y=2,
            z=4,
        )
    )
    program = Program(
        statements=[
            AssignmentStatement(
                assignments=dict(result="sum"),
                rhs_expression=CallableInvocation(
                    name="summation",
                    arguments=dict(
                        a=VariableExpr(name="x"),
                        b=VariableExpr(name="y"),
                        c=VariableExpr(name="z"),
                    ),
                ),
            )
        ],
        return_statement=ReturnStatement(
            return_values=dict(final_result=VariableExpr(name="result"))
        ),
    )
    program.evaluate(run_state)
    assert run_state.result == ResultOk(values=dict(final_result=7))


def test_summation_in_two_step_program():
    run_state = RunState(
        variables=dict(
            x=1,
            y=2,
            z=4,
        )
    )
    program = Program(
        statements=[
            AssignmentStatement(
                assignments=dict(intermediate_result="sum"),
                rhs_expression=CallableInvocation(
                    name="summation",
                    arguments=dict(
                        a=VariableExpr(name="x"),
                        b=VariableExpr(name="y"),
                    ),
                ),
            ),
            AssignmentStatement(
                assignments=dict(result="sum"),
                rhs_expression=CallableInvocation(
                    name="summation",
                    arguments=dict(
                        a=VariableExpr(name="intermediate_result"),
                        b=VariableExpr(name="z"),
                    ),
                ),
            ),
        ],
        return_statement=ReturnStatement(
            return_values=dict(final_result=VariableExpr(name="result"))
        ),
    )
    program.evaluate(run_state)
    assert run_state.result == ResultOk(values=dict(final_result=7))


def test_invocation_template():
    """Ensure that invocation templates generate correctly."""

    class ToolWithExtras(BaseCallableInputs):
        model_config = ConfigDict(extra="allow")

        x: int
        __pydantic_extra__: dict[str, int]

    class ToolWithoutExtras(BaseCallableInputs):
        x: int

    template_with_extras = ToolWithExtras.as_invocation_template("tool1")
    template_without_extras = ToolWithoutExtras.as_invocation_template("tool2")

    template_with_extras.model_validate(
        dict(
            arguments=dict(
                x=VariableExpr(name="x_src"),
            )
        )
    )

    template_without_extras.model_validate(
        dict(
            arguments=dict(
                x=VariableExpr(name="x_src"),
            )
        )
    )

    template_with_extras.model_validate(
        dict(
            arguments=dict(
                x=VariableExpr(name="x_src"),
                y=LiteralExpr(value=5),
            )
        )
    )

    with pytest.raises(ValidationError):
        template_without_extras.model_validate(
            dict(
                arguments=dict(
                    x=VariableExpr(name="x_src"),
                    y=LiteralExpr(value=5),
                )
            )
        )
