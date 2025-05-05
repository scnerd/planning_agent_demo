import decimal

from langchain_ollama import ChatOllama

from planning_agent_demo.ast.variable import PlaceholderDefinition
from planning_agent_demo.callables.self_programmer import SelfProgrammer
from planning_agent_demo.callables.summation import SummationTool


def test_that_deepseek_supports_structured_outputs():
    llm = ChatOllama(model="deepseek-r1", verbose=True, temperature=0.0)

    from pydantic import BaseModel

    class Value1(BaseModel):
        value: int

    class Value2(BaseModel):
        text: str

    class MyModel(BaseModel):
        values: dict[str, Value1 | Value2]

    llm = llm.with_structured_output(MyModel)
    messages = [
        (
            "system",
            "Generate some sample data following the specified output schema. "
            "Try to be realistic, but it doesn't matter too much.",
        ),
    ]
    result: MyModel = llm.invoke(messages)
    print(result)
    assert isinstance(result, MyModel)
    # assert result == MyModel(answer_to_life_the_universe_and_everything=42)


# def test_basic_ollama_functionality():
#     llm = ChatOllama(model=DEFAULT_MODEL, verbose=True)
#     tp = VariableDefinition
#     llm = llm.with_structured_output(tp)
#     messages = [
#         ("system",
#          "Generate some sample data following the specified output schema. "
#          "Try to be realistic, but it doesn't matter too much."),
#     ]
#     result: tp = llm.invoke(messages)
#     print(result)
#     assert isinstance(result, tp)


def test_self_programmer_directly():
    self_programming_tool = SelfProgrammer(
        name="summation agent",
        instructions="Provided two input integers a and b, compute c=a+b",
        callables=[SummationTool()],
        inputs=dict(
            a=PlaceholderDefinition(dtype="int", description="First number to add"),
            b=PlaceholderDefinition(dtype="int", description="Second number to add"),
        ),
        expected_outputs=dict(c=PlaceholderDefinition(dtype="int", description="The sum of a + b")),
    )
    assert self_programming_tool.execute(dict(a=1, b=2)).model_dump() == dict(c=decimal.Decimal(3))


def test_recursive_self_programmer():
    self_programming_tool = SelfProgrammer(
        name="summation agent",
        instructions="Provided two input integers x and y, compute z=x+y",
        callables=[SummationTool()],
        inputs=dict(
            x=PlaceholderDefinition(dtype="int", description="First number to add"),
            y=PlaceholderDefinition(dtype="int", description="Second number to add"),
        ),
        expected_outputs=dict(z=PlaceholderDefinition(dtype="int", description="The sum of x + y")),
    )
    parent_tool = SelfProgrammer(
        name="super summation agent",
        instructions="Provided four input integers--a, b, c, and d--compute e=a+b+c+d; "
        "do this as a tree, where you first add temp1=a+b, then temp2=c+d, "
        "then add temp1+temp2 to get e.",
        callables=[self_programming_tool],
        inputs=dict(
            a=PlaceholderDefinition(dtype="int", description="First number to add"),
            b=PlaceholderDefinition(dtype="int", description="Second number to add"),
            c=PlaceholderDefinition(dtype="int", description="Third number to add"),
            d=PlaceholderDefinition(dtype="int", description="Fourth number to add"),
        ),
        expected_outputs=dict(
            e=PlaceholderDefinition(dtype="int", description="The sum of a + b + c + d")
        ),
    )
    assert parent_tool.execute(dict(a=1, b=2, c=4, d=8)).model_dump() == dict(e=decimal.Decimal(15))
