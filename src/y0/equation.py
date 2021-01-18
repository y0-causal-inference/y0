import inspect
from y0.dsl import Variable, _Mathable

#Liberally borrowing from https://github.com/drhagen/parsita/blob/master/src/parsita/metaclasses.py
class VariablesDict(dict):
    def __init__(self):
        super().__init__()
        self.forward_declarations = dict()  # Stores forward declarations as they are discovered

    def __missing__(self, key):
        class_body_globals = inspect.currentframe().f_back.f_globals
        if key in class_body_globals:
            return class_body_globals[key]
        elif key in self.forward_declarations:
            return self.forward_declarations[key]
        else:
            new_forward_declaration = ForwardDeclaration(key)
            self.forward_declarations[key] = new_forward_declaration
            return new_forward_declaration

class ForwardDeclaration(Variable):
    def __init__(self, name):
        self._definition = None

    def __getattribute__(self, member):
        if member != '_definition' and self._definition is not None:
            return getattr(self._definition, member)
        else:
            return object.__getattribute__(self, member)

    def define(self, var: Variable) -> None:
        self._definition = var


def fwd() -> ForwardDeclaration:
    """Manually create a forward declaration.
    Normally, forward declarations are created automatically by the contexts.
    But they can be created manually if not in a context or if the user wants
    to avoid confusing the IDE.
    """
    return ForwardDeclaration()


class EquationMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases, **_):  # noqa: N804
        return VariablesDict()

    def __init__(cls, name, bases, dct, **_):  # noqa: N805
        super().__init__(name, bases, dct)

        # Resolve forward declarations, will raise if name not found
        for name, forward_declaration in dct.forward_declarations.items():
            obj = dct[name]
            forward_declaration._definition = Variable(name)

    def __call__(cls, *args, **kwargs):
        raise TypeError('`Equation` class cannot be instantiated. It is used to provide automatic forward declarations '
                        'for variables. Access the individual variables/equations as static attributes.')


class Equation(metaclass=EquationMeta):
    def __new__(mcs, name, bases, dct, **_):  # noqa: N804
        return super().__new__(mcs, name, bases, dct)