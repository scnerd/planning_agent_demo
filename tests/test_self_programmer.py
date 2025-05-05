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


def test_self_programmer_persistence():
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
    # Run it once, let it generate its plan
    assert self_programming_tool.execute(dict(a=1, b=2)).model_dump() == dict(c=decimal.Decimal(3))

    # Dump to disk and reload it
    self_programming_tool.save()
    pk = self_programming_tool.instance_id
    reloaded_tool = SelfProgrammer.load(pk)
    # Reloading currently throws away some data... let's just hard-code the fix for now
    # TODO: This shouldn't be needed
    reloaded_tool.callables = list(self_programming_tool.callables)

    # Run it again and make sure it works still
    # Since it already programmed itself correctly, the plan should be re-used and run VERY fast
    import time

    start = time.time()
    assert reloaded_tool.execute(dict(a=1, b=2)).model_dump() == dict(c=decimal.Decimal(3))
    end = time.time()
    assert end - start < 0.1, "Reloaded agent took way too long to run"


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
