from typing import Any, Literal

from pydantic import BaseModel, Field


class ResultOk(BaseModel):
    result_type: Literal["ok"] = Field("ok", frozen=True)
    values: dict[str, Any] = Field(..., description="The values returned by the function")


class ResultError(BaseModel):
    result_type: Literal["error"] = Field("error", frozen=True)
    error: str = Field(..., description="The error message")