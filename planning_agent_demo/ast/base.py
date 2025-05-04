from typing import Any

from pydantic import BaseModel


class BaseExpression(BaseModel):
    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__")

    def evaluate(self, run_state) -> Any:
        raise NotImplementedError("Subclasses must implement evaluate")


class BaseStatement(BaseModel):
    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__")

    def execute(self, run_state):
        raise NotImplementedError("Subclasses must implement execute")
