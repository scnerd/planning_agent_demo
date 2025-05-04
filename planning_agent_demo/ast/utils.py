import typing
from typing import Any

from pydantic import BaseModel, Field, create_model, ConfigDict

from planning_agent_demo.ast.dtype import BaseDtype
from planning_agent_demo.ast.variable import PlaceholderDefinition


class PlaceholderExtras(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    annotation: BaseDtype
    description: str


class PlaceholderDict(BaseModel):
    name: str
    placeholders: dict[str, PlaceholderDefinition]
    extras: PlaceholderExtras | None = None


def convert_pydantic_to_placeholder_dict(model: type[BaseModel]) -> PlaceholderDict:
    name = model.__name__
    placeholders = {
        field_name: PlaceholderDefinition(
            dtype=field_info.annotation,
            description=field_info.description or "",
        )
        for field_name, field_info in model.model_fields.items()
    }
    if model.model_config.get("extra") == "allow":
        # Pydantic extras is _always_ a dictionary
        extra_annotation = model.__annotations__.get("__pydantic_extra__", dict[str, Any])
        extra_annotation = typing.get_args(extra_annotation)[1]
        extra_field = model.model_fields.get('__pydantic_extra__', Field())
        extras = PlaceholderExtras(
            annotation=extra_annotation,
            description=extra_field.description or "",
        )
    else:
        extras = None

    return PlaceholderDict(
        name=name,
        placeholders=placeholders,
        extras=extras
    )


def convert_placeholder_dict_to_pydantic(placeholders: PlaceholderDict) -> type[BaseModel]:
    fields = {
        name: (placeholder.dtype.to_python_type(), Field(..., description=placeholder.description or ""))
        for name, placeholder in placeholders.placeholders.items()
    }

    config = ConfigDict(extra="forbid")

    if placeholders.extras is not None:
        config["extra"] = "allow"
        fields["__pydantic_extra__"] = (dict[str, placeholders.extras.annotation.to_python_type()], Field(..., description=placeholders.extras.description or ""))

    return create_model(
        placeholders.name,
        **fields,
        __config__=config
    )
