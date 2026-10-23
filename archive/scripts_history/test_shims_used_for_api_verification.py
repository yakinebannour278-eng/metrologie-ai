"""
Minimal, honest test double for pydantic.BaseModel/Field/model_validator -
NOT a copy of pydantic, just enough surface for src/api/schemas.py's actual
classes to be instantiated and validated with their REAL logic, since
pydantic itself cannot be installed in this sandbox (no network access,
confirmed via `pip install pydantic` failing with "No matching distribution
found"). This file is test-only scaffolding; it is not part of the shipped
project and changes no application code.
"""
import sys
import types
import inspect


class _FieldSpec:
    def __init__(self, default, **kwargs):
        self.default = default
        self.kwargs = kwargs


def Field(default=..., **kwargs):
    return _FieldSpec(default, **kwargs)


class _ValidationError(ValueError):
    pass


class BaseModel:
    def __init__(self, **data):
        annotations = {}
        for klass in reversed(type(self).__mro__):
            annotations.update(getattr(klass, "__annotations__", {}))
        for name in annotations:
            default_spec = getattr(type(self), name, None)
            if name in data:
                setattr(self, name, data[name])
            elif isinstance(default_spec, _FieldSpec):
                if default_spec.default is ...:
                    raise _ValidationError(f"Field '{name}' is required.")
                setattr(self, name, default_spec.default)
            elif default_spec is not None and not isinstance(default_spec, _FieldSpec):
                setattr(self, name, default_spec)
            else:
                setattr(self, name, None)
        for attr_name in dir(type(self)):
            attr = getattr(type(self), attr_name, None)
            if getattr(attr, "_is_model_validator_after", False):
                result = attr(self)
                if result is not None:
                    self.__dict__.update(result.__dict__)

    def model_dump(self):
        annotations = {}
        for klass in reversed(type(self).__mro__):
            annotations.update(getattr(klass, "__annotations__", {}))
        out = {}
        for name in annotations:
            val = getattr(self, name)
            out[name] = val.model_dump() if isinstance(val, BaseModel) else val
        return out


def model_validator(mode="after"):
    def decorator(fn):
        fn._is_model_validator_after = (mode == "after")
        return fn
    return decorator


module = types.ModuleType("pydantic")
module.BaseModel = BaseModel
module.Field = Field
module.model_validator = model_validator
sys.modules["pydantic"] = module

# Minimal sqlalchemy.orm.Session stub - used only as a type hint in
# prediction_service.py/main.py signatures, never actually called by our test.
sqlalchemy_mod = types.ModuleType("sqlalchemy")
sqlalchemy_orm_mod = types.ModuleType("sqlalchemy.orm")
class Session:  # noqa: N801 - matching real class name for the type hint
    pass
sqlalchemy_orm_mod.Session = Session
sqlalchemy_mod.orm = sqlalchemy_orm_mod
sys.modules["sqlalchemy"] = sqlalchemy_mod
sys.modules["sqlalchemy.orm"] = sqlalchemy_orm_mod
