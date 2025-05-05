import typing
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from planning_agent_demo.ast.callable import CallableInvocation
from planning_agent_demo.ast.dtype import BaseDtype
from planning_agent_demo.ast.variable import PlaceholderDefinition


class PlaceholderExtras(BaseModel):
    # model_config = ConfigDict(validate_assignment=True)

    annotation: BaseDtype
    description: str


# TODO: Rename
class PlaceholderDict(BaseModel):
    # name: str
    placeholders: dict[str, PlaceholderDefinition]
    extras: PlaceholderExtras | None = None

    @classmethod
    def from_pydantic(cls, model: type[BaseModel]) -> typing.Self:
        # name = model.__name__
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
            extra_field = model.model_fields.get("__pydantic_extra__", Field())
            extras = PlaceholderExtras(
                annotation=extra_annotation,
                description=extra_field.description or "",
            )
        else:
            extras = None

        return cls(
            # name=name,
            placeholders=placeholders,
            extras=extras,
        )

    def to_pydantic(self, name) -> type[BaseModel]:
        def _as_type(tp):
            if isinstance(tp, BaseDtype):
                return tp.to_python_type()
            else:
                return tp

        fields = {
            name: (
                _as_type(placeholder.dtype),
                Field(..., description=placeholder.description or ""),
            )
            for name, placeholder in self.placeholders.items()
        }

        config = ConfigDict(extra="forbid")

        if self.extras is not None:
            config["extra"] = "allow"
            fields["__pydantic_extra__"] = (
                dict[str, _as_type(self.extras.annotation)],
                Field(..., description=self.extras.description or ""),
            )

        return create_model(name, **fields, __config__=config)

    def with_values_as(self, tp) -> typing.Self:
        new_self = self.model_copy(deep=True)
        for placeholder in new_self.placeholders.values():
            placeholder.dtype = tp
        if new_self.extras is not None:
            new_self.extras.annotation = tp
        return new_self

    def to_invocation_template(self, callable_name, description) -> type[CallableInvocation]:
        from planning_agent_demo.ast.base import BaseExpression

        SpecifiedArguments = self.with_values_as(BaseDtype(root=BaseExpression)).to_pydantic(
            callable_name
        )

        class SpecifiedInvocationTemplate(CallableInvocation):
            name: Literal[callable_name] = Field(callable_name, frozen=True)  # type: ignore
            arguments: SpecifiedArguments = Field(..., description=description)  # type: ignore

        return SpecifiedInvocationTemplate
