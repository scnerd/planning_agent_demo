import traceback
from typing import Annotated, Any, Literal

from pydantic import Field

import planning_agent_demo
from planning_agent_demo.ast.base import BaseExpression, BaseStatement


class VariableExpr(BaseExpression):
    expr_type: Literal["variable"] = Field("variable", frozen=True)
    name: str = Field(..., description="The name of the variable")

    def __str__(self):
        return self.name

    def evaluate(self, run_state):
        return run_state.variables[self.name]


class LiteralExpr(BaseExpression):
    expr_type: Literal["literal"] = Field("literal", frozen=True)
    value: Any = Field(..., description="The literal value")

    def __str__(self):
        return repr(self.value)

    def evaluate(self, run_state):
        return self.value


class CallableInvocation(BaseExpression):
    expr_type: Literal["func_call"] = Field("func_call", frozen=True)

    name: str
    arguments: dict[str, "RhsExpression"]

    def __str__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.arguments.items())})"

    def evaluate(self, run_state: "planning_agent_demo.ast.run_state.RunState") -> Any:
        callable_instance = run_state.callables[self.name]
        args = {key: value.evaluate(run_state) for key, value in self.arguments.items()}
        args = callable_instance.inputs_type(**args)
        result = callable_instance.execute(args)
        return result.model_dump()


RhsExpression = Annotated[
    VariableExpr | LiteralExpr | CallableInvocation, Field(discriminator="expr_type")
]


class AssignmentStatement(BaseStatement):
    stmt_type: Literal["invocation"] = Field("invocation", frozen=True)
    assignments: dict[str, str] = Field(
        ...,
        description="Dictionary mapping current-scope variables to assign to child-scope return names",
    )
    rhs_expression: RhsExpression = Field(..., description="The expression to be evaluated")

    def __str__(self):
        assignment_txt = ", ".join(f"{k} <- {v}" for k, v in self.assignments.items())
        return f"({assignment_txt}) = {self.rhs_expression}"

    def execute(self, run_state):
        result = self.rhs_expression.evaluate(run_state)
        for k, v in self.assignments.items():
            run_state.variables[k] = result[v]


class ReturnStatement(BaseStatement):
    stmt_type: Literal["return"] = Field("return", frozen=True)
    return_values: dict[str, RhsExpression] = Field(
        ..., description="The names to return the the expressions to return in them"
    )

    def __str__(self):
        return f"return {', '.join(f'{k}={v}' for k, v in self.return_values.items())}"

    def execute(self, run_state):
        from planning_agent_demo.ast.result import ResultOk

        run_state.result = ResultOk(
            values={k: v.evaluate(run_state) for k, v in self.return_values.items()}
        )


NonterminalStatement = Annotated[AssignmentStatement, Field(discriminator="stmt_type")]
TerminalStatement = Annotated[ReturnStatement, Field(discriminator="stmt_type")]


# Statement = Annotated[Union[NonterminalStatement, TerminalStatement], Field(discriminator="stmt_type")]


class Program(BaseExpression):
    statements: list[NonterminalStatement] = Field(
        ..., description="The list of statements to be executed"
    )
    return_statement: TerminalStatement = Field(
        ..., description="The return statement to be executed"
    )

    def __str__(self):
        return (
            "\n".join(str(statement) for statement in self.statements)
            + "\n\n"
            + str(self.return_statement)
        )

    def evaluate(self, run_state):
        try:
            for statement in self.statements:
                statement.execute(run_state)
                if run_state.result is not None:
                    return
            self.return_statement.execute(run_state)
        except Exception:
            message = traceback.format_exc()
            from planning_agent_demo.ast.result import ResultError

            run_state.result = ResultError(error=message)
