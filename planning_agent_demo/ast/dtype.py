import decimal

from pydantic import RootModel, model_validator


class BaseDtype(RootModel):
    root: type

    @model_validator(mode='before')
    def accept_common_strings_as_aliases(cls, v):
        if isinstance(v, str):
            v = {
                "str": str,
                "int": decimal.Decimal,
                "float": decimal.Decimal,
                "decimal": decimal.Decimal,
                "bool": bool,
            }.get(v, v)
        return v

    # def __str__(self):
    #     raise NotImplementedError("Subclasses must implement __str__")

    def to_python_type(self):
        return self.root


# class StringDtype(BaseDtype):
#     tp: type = str
#
#
# class NumberDtype(BaseDtype):
#     tp: type = decimal.Decimal
#
#
# class BoolDtype(BaseDtype):
#     tp: type = bool
