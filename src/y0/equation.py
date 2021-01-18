import inspect
from y0.dsl import Variable, _Mathable

#Somewhat based on  https://github.com/drhagen/parsita/blob/master/src/parsita/metaclasses.py
#TODO: Inheret from ABC to ensure that 'eq' exists.  
#TODO: Make Equation a subclass of _Mathable.  This will require monkeying with _Mathable so the metatypes line up properly.
#TODO: Can you grab the body of a class and wrap it in an assignment?  If so, we could remove a bit of boilerplate!

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
        
    def __repr__(cls): return str(vars(cls)["eq"])
        
    def _repr_latex_(cls): return f'${vars(cls)["eq"].to_latex()}$'
    
    def __call__(cls, *args, **kwargs):
        raise TypeError('`Equation` class cannot be instantiated. It is used to provide automatic forward declarations '
                        'for variables. Access the individual variables/equations as static attributes.')

class Equation(metaclass=EquationMeta): pass
  