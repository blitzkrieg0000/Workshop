    from types import FunctionType
    from functools import wraps

    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            print('{!r} executing'.format(method.__name__))
            return method(*args, **kwargs)
        return wrapped

    class MetaClass(type):
        def __new__(meta, classname, bases, classDict):
            newClassDict = {}
            for attributeName, attribute in classDict.items():
                if isinstance(attribute, FunctionType):
                    attribute = wrapper(attribute)
                newClassDict[attributeName] = attribute
            return type.__new__(meta, classname, bases, newClassDict)

    def with_metaclass(meta):
        """ Create an empty class with the supplied bases and metaclass. """
        return type.__new__(meta, "TempBaseClass", (object,), {})


    # Inherit metaclass from a dynamically-created base class.
    class MyClass(with_metaclass(MetaClass)):
        @staticmethod
        def a_static_method():
            pass

        @classmethod
        def a_class_method(cls):
            pass

        def a_method(self):
            print("main")

    instance = MyClass()
    instance.a_static_method()  # Not decorated.
    instance.a_class_method()   # Not decorated.
    instance.a_method()         # -> 'a_method' executing
