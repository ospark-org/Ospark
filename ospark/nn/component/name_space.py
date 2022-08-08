class PrefixNameMeta(type):

    is_instanced = False
    instance_class = None

    def __call__(mcls, *args, **kwargs):
        if mcls.is_instanced:
            obj = mcls.instance_class
        else:
            obj = super().__call__(*args, **kwargs)
            mcls.instance_class = obj
            mcls.is_instanced = True
        return obj


class PrefixName(metaclass=PrefixNameMeta):

    _prefix_name = ""

    def __add__(self, name: str):
        self._prefix_name += name
        return self

    def __getitem__(self, key):
        self._prefix_name = self._prefix_name[key]
        return self

    @property
    def name(self):
        return self._prefix_name


class NameSpace:

    def __init__(self, name: str):
        self._name   = name
        self._prefix = PrefixName()

    def __enter__(self):
        self._prefix + self._name + "/"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._prefix = self._prefix[:-(len(self._name) + 1)]