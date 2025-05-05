import decimal
import textwrap
from functools import cache
from typing import Literal, Union, ClassVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from planning_agent_demo.ast.callable import CallableDefinition
from planning_agent_demo.ast.expression import (
    AssignmentStatement,
    Program,
    ReturnStatement,
    VariableExpr,
    CallableInvocation,
)
from planning_agent_demo.ast.result import ResultError, ResultOk
from planning_agent_demo.ast.utils import PlaceholderDict
from planning_agent_demo.ast.variable import PlaceholderDefinition
from planning_agent_demo.callables.base import BaseCallable, BaseStatefulCallable

# DEFAULT_MODEL = "llama3.2"
DEFAULT_MODEL = "deepseek-r1"


def str_choice(choices: list[str]) -> type:
    return Union[*[Literal[v] for v in choices]]


# str_var_name = constr(pattern=r"[\w_][\w\d_]*")
str_var_name = str


class ProgramOverview(BaseModel):
    initial_thoughts: str = Field(
        ...,
        description="A high-level summary (one or two sentences) of how you will approach writing a correct program",
    )
    detailed_thoughts: str = Field(
        ...,
        description="In-depth details about whatever you'll need to figure out to write the program",
    )
    concluding_thoughts: str = Field(
        ..., description="Any final thoughts before beginning to write the program"
    )


class ProgramRoughStep(BaseModel):
    step_description: str = Field(
        ...,
        description="A short description of what function we will probably call and what we expect it to do",
    )
    expected_output_variable_names: list[str_var_name] = Field(
        ...,
        description="The names of any variables expected to be assigned/created from the values returned by this function call; for example `x` or `total_amount`. Variable names should be `snake_case`.",
    )


class ProgramRoughPlan(BaseModel):
    implementation_steps: list[ProgramRoughStep]


class VariableArgument(BaseModel):
    variable_name: str_var_name

    def __str__(self):
        return self.variable_name

    def to_expression(self) -> VariableExpr:
        return VariableExpr(name=self.variable_name)


class LiteralArgument(BaseModel):
    literal_value: str | decimal.Decimal | bool

    # def __str__(self):
    #     return repr(self.constant)
    #
    # def to_expression(self) -> LiteralExpr:
    #     return LiteralExpr(value=self.constant)


class ProgramFormalStep(BaseModel):
    function: str
    arguments: dict[str, VariableArgument | LiteralArgument]
    result_assignments: dict[str, str]

    def __str__(self):
        assignments = ", ".join(f"{k} <- {v}" for k, v in self.result_assignments.items())
        arguments = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return f"({assignments}) = {self.function}({arguments})"

    @classmethod
    def create_specified_formal_step(
        cls,
        function_name: str,
        args_type: type[BaseModel],
        existing_variables: list[str],
        returned_variables: list[str],
    ):
        class ConstrainedVariableArgument(VariableArgument):
            variable_name: str_choice(existing_variables)

        args_type = (
            PlaceholderDict.from_pydantic(args_type)
            .with_values_as(ConstrainedVariableArgument | LiteralArgument)
            .to_pydantic(name="Arguments")
        )

        class SpecifiedProgramFormalStep(ProgramFormalStep):
            function: Literal[function_name] = Field(function_name, frozen=True)
            arguments: args_type = Field(
                ...,
                description="The arguments to pass into the function at run time. Keys are the parameter names in the function, and the values are the expressions to evaluate to pass to those parameters at runtime.",
            )
            result_assignments: dict[str_var_name, str_choice(returned_variables)] = Field(
                ...,
                description="A map from local variable names to return-value names to bind to those local variables. E.g., {'x': 'result'} would take the 'result' value returned by the function call and bind it to a variable named 'x' in the local scope.",
            )

        return SpecifiedProgramFormalStep

    def to_statement(self) -> AssignmentStatement:
        return AssignmentStatement(
            assignments=self.result_assignments,
            rhs_expression=CallableInvocation(
                name=self.function,
                arguments={k: v.to_expression() for k, v in self.arguments.items()},
            ),
        )


class ProgramReturnStep(BaseModel):
    return_values: dict[str_var_name, str_var_name] = Field(
        ...,
        description="The variables to return to the calling function; the key is the name to return, and must be one of the pre-defined output names; the value must be the current-scope variable whose value should be returned under that name.",
    )

    @classmethod
    def create_specified_return_step(
        cls, existing_variables: list[str], expected_outputs: list[str]
    ):
        class SpecifiedProgramReturnStep(ProgramReturnStep):
            return_values: dict[str_choice(expected_outputs), str_choice(existing_variables)] = (
                Field(
                    ...,
                    description="The variables to return to the calling function; the key is the name to return, and must be one of the pre-defined output names; the value must be the current-scope variable whose value should be returned under that name.",
                )
            )

        return SpecifiedProgramReturnStep

    def to_statement(self) -> ReturnStatement:
        return ReturnStatement(
            return_values={k: VariableExpr(name=v) for k, v in self.return_values.items()}
        )


@cache
def llm():
    return ChatOllama(model=DEFAULT_MODEL, verbose=True, temperature=0.0)


def structured_llm_call[O: BaseModel](
    output_model: type[O], generate_model: type[BaseModel] | None = None, *, messages
) -> O:
    if generate_model is None:
        generate_model = output_model
    structured_llm = llm().with_structured_output(generate_model)
    result = structured_llm.invoke(messages)
    result = output_model(**result.model_dump())
    return result


class SelfProgrammer(BaseStatefulCallable):
    type_prefix: ClassVar[str] = "self_programmer"

    name: str
    instructions: str
    callables: list[BaseCallable]
    inputs: dict[str, PlaceholderDefinition]
    expected_outputs: dict[str, PlaceholderDefinition]
    program: Program | None = None

    _input_model: type[BaseModel] | None = None

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
    def _inputs_definition(self):
        return PlaceholderDict(
            # name=self.name,
            placeholders=self.inputs,
            extras=None,
        )

    @property
    def _outputs_definition(self):
        return PlaceholderDict(
            # name=self.name,
            placeholders=self.expected_outputs,
            extras=None,
        )

    @property
    def invocation_template(self) -> type[CallableInvocation]:
        return self._inputs_definition.to_invocation_template(self.instructions)

    @property
    def inputs_type(self):
        if self._input_model is None:
            self._input_model = self._inputs_definition.to_pydantic("SelfProgrammerInputs")
        return self._input_model

    @property
    def result_type(self):
        return self._outputs_definition.to_pydantic("SelfProgrammerOutputs")

    def _generate_plan(self, arguments: BaseModel) -> Program:
        arguments = self.inputs_type.model_validate(arguments)

        print("Generating program")
        tool_descriptions = "\n".join(
            f"- `{fn.definition.name}`: {fn.definition.description}" for fn in self.callables
        )

        messages = [
            (
                "system",
                textwrap.dedent("""
                        You are a helpful programmer who writes programs to solve problems for others.

                        Given the requested task, write the AST for an application that will solve the specified problem.

                        Your code will consist entirely of function calls and assigning the return values.
                        For example, a pretty version of a program might look like:
                        ```
                        (x <- result) = function(param1=y, param2=10)
                        ```
                        In this example, we've called `function` passing in a local variable `y` for `param1` and a literal value 10 for `param2`.
                        This function returns a single named value called `result`, which we store in a new local variable named `x`.

                        At the end, you'll have the chance to specify which variables should be returned to satisfy the overall function requirements.
                        For example, if we are supposed to return something called `final_value`, you will be allowed to say:
                        ```
                        return final_value=x
                        ```
                        which means that the calling function will receive a value named `final_value` which will have whatever `x` had in our scope.

                        The user will tell you what function they want you to write and what functions you'll have available to call. 
                        """).strip(),
            ),
            ("human", self.instructions),
            ("human", f"Here are the tools you have available:\n{tool_descriptions}"),
        ]

        print("Generating initial ideas...")
        plan_overview: ProgramOverview = structured_llm_call(ProgramOverview, messages=messages)
        messages.extend(
            [
                ("assistant", plan_overview.initial_thoughts),
                ("assistant", plan_overview.detailed_thoughts),
                ("assistant", plan_overview.concluding_thoughts),
                (
                    "assistant",
                    "Ok, now I need to plan out what my function will look like, one line of code at a time. I need to remember that each line will be exactly one function call.",
                ),
            ]
        )

        print("Converting ideas into logical plan...")
        plan_rough_draft: ProgramRoughPlan = structured_llm_call(
            ProgramRoughPlan,
            messages=messages,
        )
        numbered_outline = "\n".join(
            f"{i}. {step.step_description.rstrip('.')}. This will generate the following variables: {step.expected_output_variable_names}"
            for i, step in enumerate(plan_rough_draft.implementation_steps, 1)
        )
        messages.append(("assistant", numbered_outline))

        formal_steps: list[ProgramFormalStep] = []
        available_variables: set[str] = set(self.inputs)

        for i, step in enumerate(plan_rough_draft.implementation_steps, 1):
            available_tool_calls = [
                ProgramFormalStep.create_specified_formal_step(
                    function_name=fn.definition.name,
                    args_type=fn.inputs_type,
                    existing_variables=list(available_variables),
                    returned_variables=list(fn.definition.returns),
                )
                for fn in self.callables
            ]
            tool_call_type = Union[*available_tool_calls]

            print(f"Generating function call for step {i}...")
            messages.append(("assistant", f"Let's finish defining step {i}"))
            formal_step = structured_llm_call(ProgramFormalStep, tool_call_type, messages=messages)
            formal_steps.append(formal_step)
            available_variables.update(formal_step.result_assignments.keys())
            messages.append(("assistant", str(formal_step)))

        print("Generating return definition...")
        messages.append(
            (
                "assistant",
                textwrap.dedent(f"""
                    Now, let's define what values should be returned.

                    I'm expected to return the following values: {list(self.expected_outputs)}.

                    I have the following variables available: {list(available_variables)}

                    I just need to assign those variables to the expected output names.
                    """).strip(),
            )
        )
        return_step = structured_llm_call(
            ProgramReturnStep,
            ProgramReturnStep.create_specified_return_step(
                existing_variables=list(available_variables),
                expected_outputs=list(self.expected_outputs),
            ),
            messages=messages,
        )

        print("Finalizing...")
        return Program(
            statements=[step.to_statement() for step in formal_steps],
            return_statement=return_step.to_statement(),
        )

    def _run_plan(self, arguments: BaseModel) -> BaseModel:
        print("Executing plan...")
        from planning_agent_demo.ast.run_state import RunState

        run_state = RunState(available_callables=self.callables, variables=arguments.model_dump())

        self.program.evaluate(run_state)
        print(f"{run_state.result=}")

        match run_state.result:
            case ResultError(error=msg):
                raise RuntimeError(f"Program failed to execute successfully: {msg}")
            case ResultOk(values=data):
                return self.result_type.model_validate(data)

    def execute(self, arguments: BaseModel) -> BaseModel:
        if not isinstance(arguments, BaseModel):
            arguments = self.inputs_type(**arguments)

        if self.program is None:
            self.program = self._generate_plan(arguments)

            print(f"Program generated:\n```\n{self.program}\n```")

        return self._run_plan(arguments)
