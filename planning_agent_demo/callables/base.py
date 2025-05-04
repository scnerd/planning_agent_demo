import abc
import shelve
import uuid
from typing import ClassVar, Self, Literal

from pydantic import BaseModel, create_model, Field

from planning_agent_demo.ast.base import BaseExpression
from planning_agent_demo.ast.callable import CallableDefinition, CallableInvocation, PlaceholderDefinition
from planning_agent_demo.ast.dtype import BaseDtype
from planning_agent_demo.ast.utils import convert_placeholder_dict_to_pydantic, PlaceholderDict, \
    convert_pydantic_to_placeholder_dict


class BaseCallableInputs(BaseModel):
    @classmethod
    def as_parameters(cls) -> PlaceholderDict:
        return convert_pydantic_to_placeholder_dict(cls)

    @classmethod
    def as_allow_extra_parameters(cls) -> bool:
        return cls.model_config["extra"] == "allow"

    @classmethod
    def as_invocation_template(cls, callable_name: str) -> type[CallableInvocation]:
        """An invocation template is a pydantic model which an LLM can generate to specify a static (code) way of calling a function.

        The template is specific to the function being called, allowing for stronger guarantees around required inputs, extras, types, etc.

        The template does _not_ contain the values being passed, either before or after the LLM call, it only defines the
        expressions that should be evaluated at runtime to obtain a value
        """

        parameter_dict = cls.as_parameters()
        for placeholder in parameter_dict.placeholders.values():
            placeholder.dtype = BaseDtype(root=BaseExpression)
        if parameter_dict.extras is not None:
            parameter_dict.extras.annotation = BaseDtype(root=BaseExpression)

        SpecifiedArguments = convert_placeholder_dict_to_pydantic(parameter_dict)

        class SpecifiedInvocationTemplate(CallableInvocation):
            name: Literal[callable_name] = Field(callable_name, frozen=True)  # type: ignore
            arguments: SpecifiedArguments = Field(..., description=cls.__doc__ or "The arguments to be passed to the callable")  # type: ignore

        return SpecifiedInvocationTemplate


class BaseCallableOutputs(BaseModel):
    @classmethod
    def as_returns(cls) -> dict[str, PlaceholderDefinition]:
        return {
            field_name: PlaceholderDefinition(
                dtype=field_info.annotation.__name__,
                description=field_info.description or "",
            )
            for field_name, field_info in cls.model_fields.items()
        }


class BaseCallable[I: BaseModel, O: BaseModel](BaseModel, abc.ABC):
    __register_callable__: ClassVar[bool] = False
    __registry__: ClassVar[list[Self]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__register_callable__:
            cls.__registry__.append(cls())

    @property
    def definition(self) -> CallableDefinition:
        raise NotImplementedError()

    @property
    def invocation_template(self) -> type[CallableInvocation]:
        raise NotImplementedError()

    @property
    def inputs_type(self) -> type[I]:
        raise NotImplementedError()

    @property
    def result_type(self) -> type[O]:
        raise NotImplementedError()

    def execute(self, arguments: I) -> O:
        raise NotImplementedError()


class SimpleCallable[I: BaseCallableInputs, O: BaseCallableOutputs](BaseCallable[I, O], abc.ABC):
    __register_callable__: ClassVar[bool] = False

    name: ClassVar[str]
    description: ClassVar[str]
    inputs: ClassVar[type[BaseCallableInputs]]
    outputs: ClassVar[type[BaseCallableOutputs]]

    @property
    def definition(self) -> CallableDefinition:
        parameters = self.inputs.as_parameters()
        return CallableDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters.placeholders,
            allow_extra_parameters=parameters.extras is not None,
            returns=self.outputs.as_returns(),
        )

    @property
    def invocation_template(self) -> type[CallableInvocation]:
        return self.inputs.as_invocation_template(self.name)

    @property
    def inputs_type(self) -> type[BaseCallableInputs]:
        return self.inputs

    @property
    def result_type(self) -> type[BaseCallableOutputs]:
        return self.outputs


class BaseStatefulCallable(BaseCallable, abc.ABC):
    __register_callable__: ClassVar[bool] = False

    type_prefix: ClassVar[str]
    instance_id: uuid.UUID

    def save(self):
        # Use shelve to persist the JSON version of this model to disk
        with shelve.open("callable_instances") as db:
            db[f"{self.type_prefix}/{self.instance_id}"] = self.model_dump()

    @classmethod
    def load(cls, instance_id: uuid.UUID) -> Self:
        # Use shelve to load the JSON version of this model from disk
        with shelve.open("callable_instances") as db:
            return cls.model_validate(db[f"{cls.type_prefix}/{instance_id}"])
