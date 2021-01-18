import inspect
from y0.dsl import Variable, _Mathable

#Liberally borrowing from https://github.com/drhagen/parsita/blob/master/src/parsita/metaclasses.py
class _MakeVariables(dict):
    def __missing__(self, key):
        class_body_globals = inspect.currentframe().f_back.f_globals
        if key in class_body_globals:
            return class_body_globals[key]
        else:
            return Variable(key)

class EquationMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases, **_):  # noqa: N804
        return _MakeVariables()
    
    def _default_value(cls):
        keys = [k for k in vars(cls).keys() 
               if not k.startswith("_")]
        if len(keys) > 0:
            return vars(cls)[keys[0]]
        else:
            raise ValueError("No default Value")
    
    def __repr__(cls):
        try: return str(cls._default_value())
        except ValueError: return super().__repr__()     
        
    def _repr_latex_(cls):
        try: return f'${cls._default_value().to_latex()}$'
        except ValueError: return super().__repr__()
    
    def __call__(cls, *args, **kwargs):
        raise TypeError('`Equation` class cannot be instantiated. It is used to provide automatic forward declarations '
                        'for variables. Access the individual variables/equations as static attributes.')

class Equation(metaclass=EquationMeta):
    def __new__(mcs, name, bases, dct, **_):  # noqa: N804
        return super().__new__(mcs, name, bases, dct)
  