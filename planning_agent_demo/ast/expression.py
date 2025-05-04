from typing import Any, Literal, Union, Annotated

from pydantic import BaseModel, Field

from planning_agent_demo.ast.base import BaseExpression, BaseStatement
from planning_agent_demo.ast.callable import CallableInvocation


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


RhsExpression = Annotated[Union[VariableExpr, LiteralExpr, CallableInvocation], Field(discriminator="expr_type")]



class AssignmentStatement(BaseStatement):
    stmt_type: Literal["invocation"] = Field("invocation", frozen=True)
    assignments: dict[str, str] = Field(..., description="Dictionary mapping current-scope variables to assign to child-scope return names")
    rhs_expression: RhsExpression = Field(..., description="The expression to be evaluated")

    def __str__(self):
        return f"{', '.join(f'{k}={v}' for k, v in self.assignments.items())} = {self.rhs_expression}"

    def execute(self, run_state):
        result = self.rhs_expression.evaluate(run_state)
        for k, v in self.assignments.items():
            run_state.variables[k] = result[v]


class ReturnStatement(BaseStatement):
    stmt_type: Literal["return"] = Field("return", frozen=True)
    return_values: dict[str, RhsExpression] = Field(..., description="The names to return the the expressions to return in them")

    def __str__(self):
        return f"return {', '.join(f'{k}={v}' for k, v in self.return_values.items())}"

    def execute(self, run_state):
        from planning_agent_demo.ast.common import ResultOk
        run_state.result = ResultOk(values={k: v.evaluate(run_state) for k, v in self.return_values.items()})


NonterminalStatement = Annotated[Union[AssignmentStatement], Field(discriminator="stmt_type")]
TerminalStatement = Annotated[Union[ReturnStatement], Field(discriminator="stmt_type")]
# Statement = Annotated[Union[NonterminalStatement, TerminalStatement], Field(discriminator="stmt_type")]


class Program(BaseExpression):
    statements: list[NonterminalStatement] = Field(..., description="The list of statements to be executed")
    return_statement: TerminalStatement = Field(..., description="The return statement to be executed")

    def __str__(self):
        return "\n".join(str(statement) for statement in self.statements) + "\n\n" + str(self.return_statement)

    def evaluate(self, run_state):
        try:
            for statement in self.statements:
                statement.execute(run_state)
                if run_state.result is not None:
                    return
            self.return_statement.execute(run_state)
        except Exception as e:
            message = str(e)
            from planning_agent_demo.ast.common import ResultError
            run_state.result = ResultError(error=message)



