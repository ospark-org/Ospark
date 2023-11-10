from inspect import signature, Signature
from typing import List, Tuple, Any, Optional
import copy
import click


class AutoClick(type):
    classes        = {}
    cli            = click.group(lambda: None)
    default_values = {}

    def __new__(mcls, *args, **kwargs):
        cls = super().__new__(mcls, *args, **kwargs)

        copy_cls     = copy.copy(cls)
        default_dict = mcls.classes.setdefault(copy_cls.__name__, {})

        default_dict["normal_cls"] = copy_cls
        default_values             = {}

        sig_para, str_sig = mcls.get_signature(cls=cls)

        for para_name, para_type in str_sig:
            if "self" == para_name:
                continue

            para_types, _, required, default_value = mcls.get_para_type(type_string=para_type, sig_para=sig_para,
                                                                        para_name=para_name)

            default_values[para_name] = default_value

            cls = click.option(f'-{"".join([word[0] for word in para_name.split("_")])}',
                               f'--{"-".join(para_name.split("_"))}',
                               required=required,
                               type=para_types[0] if type(para_types) == type else str,
                               default=default_value)(cls)
        cls = mcls.cli.command()(cls)

        mcls.default_values.setdefault(copy_cls.__name__, default_values)
        default_dict["click_cls"] = cls
        return default_dict["normal_cls"]

    def main_process(cls):
        raise NotImplementedError()

    @staticmethod
    def get_signature(cls) -> Tuple[Signature, List[List[str]]]:
        sig_para = signature(cls.__dict__["__init__"])
        # str_sig   = [para_type_pair.replace(" ", "").split(":") if ":" in para_type_pair else [para_type_pair, ""]
        #              for para_type_pair in str(sig_para)[1:-1].split(", ")]

        count = 0
        _paras = ""
        for string in str(sig_para)[1:-1]:
            if string == "[" or string == "{" or string == "(":
                count += 1
                _paras += string
            elif string == "]" or string == "}" or string == ")":
                count -= 1
                _paras += string
            elif count != 0 and string == ",":
                _paras += "$"
            elif count != 0 and string == ":":
                _paras += "%"
            else:
                _paras += string

        result = []
        for para in _paras.replace(" ", "").split(",")[1:]:
            para = para.replace("$", ",")
            para = para.split(":")
            para[1] = para[1].replace("%", ":")
            result.append(para)

        return sig_para, result

    @staticmethod
    def get_para_type(type_string: str,
                      sig_para: Signature,
                      para_name: str,
                      para_value: Optional[str]=None) -> Tuple[List[type], type, bool, Any]:
        outer_typing  = sig_para.parameters[para_name].annotation
        required      = False
        if "=" not in type_string:
            required      = True
            default_value = para_value
        else:
            if "Optional" in type_string:
                type_string, default_value = type_string.replace(" ", "").replace("'", "").split("=")
                type_string = type_string.replace("Optional", "")[1:-1]
            else:
                type_string, default_value = type_string.replace(" ", "").replace("'", "").split("=")

            default_value = default_value if para_value is None else para_value

        if type(default_value) == str:
            default_value = AutoClick.wrap_typing(input_typing=type_string, default_value=default_value)

        if "List" == type_string[:4]:
            inner_typing = type_string.replace(" ", "")[5:-1].split(",")
            outer_typing = list
        elif "Dict" == type_string[:4]:
            inner_typing = type_string.replace(" ", "")[5:-1].split(",")
            outer_typing = dict
        elif "Tuple" == type_string[:5]:
            inner_typing = type_string.replace(" ", "")[6:-1].split(",")
            outer_typing = tuple
        else:
            inner_typing = [type_string]

        para_types = []
        for typing in inner_typing:
            if typing == "int":
                para_type = int
            elif typing == "float":
                para_type = float
            elif typing == "bool":
                para_type = bool
            elif typing == "str":
                para_type = str
            else:
                para_type = typing
            para_types.append(para_type)

        return para_types, outer_typing, required, default_value

    @classmethod
    def process_value(cls, typing: str, values: str):
        results = []
        count = 0
        temp_sequence = []

        for value in values[1: -1].replace(" ", "").split(","):
            if "[" in value or "{" in value or "(" in value:
                count += value.count("[") + value.count("(") + value.count("{")
                temp_sequence += [value]
            elif "]" in value or "}" in value or ")" in value:
                count -= value.count("]") + value.count(")") + value.count("}")
                temp_sequence += [value]
                if count == 0:
                    results += [",".join(temp_sequence)]
                    temp_sequence = []
                else:
                    continue
            elif count > 0:
                temp_sequence += [value]
            else:
                results += [value]

        if typing == "Dict":
            results = [pairs.split(":", 1) for pairs in results]
        # else:
        #     results = [value for value in values[1:-1].replace(" ", "").split(",")]
        return results

    @staticmethod
    def wrap_typing(input_typing: str, default_value: str):
        def wrap(typing: str, values: Any) -> Any:
            if values is None or values == "None":
                return None

            result = None
            if typing == "Dict":
                result = {key: value for key, value in values}
            elif typing == "List":
                result = [value for value in values]
            elif typing == "Tuple":
                result = tuple(values)
            elif typing == "Set":
                result = {value for value in values}
            elif typing == "int":
                result = int(values)
            elif typing == "float":
                result = float(values)
            elif typing == "str":
                result = str(values)
            elif typing == "bool":
                if values == "True" or values == "true":
                    result = True
                else:
                    result = False
            else:
                pass
            if result is None:
                raise KeyError(f"typing: {typing} is not support.")
            return result

        if "[" in input_typing and "]" in input_typing:
            typing, inner = input_typing.split("[", 1)

            values = AutoClick.process_value(typing=typing, values=default_value)
            if typing in ("Dict", "Tuple"):
                results = []
                if typing == "Tuple":
                    inner_types = inner[:-1].replace(" ", "").split(",")
                    if len(inner_types) != len(values):
                        inner_types *= len(values)
                    for inner_type, inner_value in zip(inner_types, values):
                        results.append(AutoClick.wrap_typing(input_typing=inner_type, default_value=inner_value))
                else:
                    results = []
                    for value in values:
                        inner_results = []
                        for inner_type, inner_value in zip(inner[:-1].replace(" ", "").split(",", 1), value):
                            inner_results.append(AutoClick.wrap_typing(input_typing=inner_type,
                                                                       default_value=inner_value))
                        results.append(inner_results)

            else:
                results = []
                for value in values:
                    results.append(AutoClick.wrap_typing(input_typing=inner[:-1], default_value=value))

            results = AutoClick.wrap_typing(input_typing=typing, default_value=results)

            return results
        else:
            values = wrap(input_typing, default_value)
            return values

    def __call__(cls, *args, **kwargs):

        if len(args) == 0 and len(kwargs) == 0:
            _cls = cls.classes[cls.__name__]["click_cls"]
            _cls.__call__()
        else:
            sig_para, str_sig = cls.get_signature(cls=cls)

            if len(args) != 0:
                paras = {str_sig[i][0]: value for i, value in enumerate(args)}
                cls.default_values[cls.__name__].update(paras)
            cls.default_values[cls.__name__].update(kwargs)

            kwargs = cls.default_values[cls.__name__]
            obj    = super().__call__(**kwargs)

            for i, value in enumerate(args):
                para_name, para_type = str_sig[i]
                para_types, outer_type, _, value = cls.get_para_type(type_string=para_type, sig_para=sig_para,
                                                                     para_name=para_name, para_value=value)

                setattr(obj, f"_{para_name}", value)

            for key, value in kwargs.items():
                para_type = list(filter(lambda input: input[0] == key, str_sig))

                para_types, outer_type, _, value = cls.get_para_type(type_string=para_type[0][1], sig_para=sig_para,
                                                                     para_name=key, para_value=value)

                if value is not None or obj.__dict__.get(f"_{key}") is None:
                    setattr(obj, f"_{key}", value)
        obj.main_process()
        return obj