from inspect import signature, Signature
from typing import List, Tuple
import copy
import click


class AutoClick(type):
    classes = {}
    cli     = click.group(lambda :None)

    def __new__(mcls, *args, **kwargs):
        cls = super().__new__(mcls, *args, **kwargs)
        copy_cls = copy.copy(cls)
        default_dict = mcls.classes.setdefault(copy_cls.__name__, {})
        default_dict["normal_cls"] = copy_cls

        sig_para, str_sig = mcls.get_signature(cls=cls)

        for para_name, para_type in str_sig:
            if "self" == para_name:
                continue

            para_types, _, required = mcls.get_para_type(type_string=para_type, sig_para=sig_para, para_name=para_name)

            cls = click.option(f'-{"".join([word[0] for word in para_name.split("_")])}',
                               f'--{"-".join(para_name.split("_"))}',
                               required=required,
                               type=para_types[0] if type(para_types) == type else str,
                               default=None)(cls)
        cls = mcls.cli.command()(cls)

        default_dict["click_cls"] = cls
        return default_dict["normal_cls"]

    def main_process(cls):
        raise NotImplementedError()

    @staticmethod
    def get_signature(cls) -> Tuple[Signature, List[List[str]]]:
        sig_para  = signature(cls.__dict__["__init__"])
        str_sig   = [para_type_pair.replace(" ", "").split(":") if ":" in para_type_pair else [para_type_pair, ""]
                     for para_type_pair in str(sig_para)[1:-1].split(", ")]
        return sig_para, str_sig

    @staticmethod
    def string_transformation(string: str, para_types: List[type], outer_type: type):
        if string is None or string == "None":
            return None
        elif string == "True":
            return True
        elif string == "False":
            return False
        else:
            pass

        if outer_type == list:
            result = outer_type(para_types[0](value) for value in string.strip("[]").replace(" ", "").split(","))
        elif outer_type == dict:
            result = outer_type([(para_types[i](value) for i, value in enumerate(values.split(":")))
                                 for values in string.strip("{}").replace(" ", "").split(",")])
        elif outer_type == tuple:
            result = outer_type([para_types[i](value) for i, value in enumerate(string.strip("()").replace(" ", "").split(","))])
        else:
            result = para_types[0](string) if para_types[0] is not None else outer_type(string)
        return result
    @staticmethod
    def get_para_type(type_string: str, sig_para: Signature, para_name: str) -> Tuple[List[type], type, bool]:
        outer_typing = sig_para.parameters[para_name].annotation

        if "Optional" not in type_string:
            required = True
        else:
            required = False
            type_string = type_string.replace(" ", "").replace("Optional", "").replace("None", "").replace("=", "").strip("[]")

        if "List" in type_string:
            inner_typing = type_string.replace(" ", "").replace("List", "").strip("[]").split(",")
            outer_typing = list
        elif "Dict" in type_string:
            inner_typing = type_string.replace(" ", "").replace("Dict", "").strip("[]").split(",")
            outer_typing = dict
        elif "Tuple" in type_string:
            inner_typing = type_string.replace(" ", "").replace("Tuple", "").strip("[]").split(",")
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
        return para_types, outer_typing, required

    def __call__(cls, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            _cls = cls.classes[cls.__name__]["click_cls"]
            _cls.__call__()
        else:
            sig_para, str_sig = cls.get_signature(cls=cls)

            obj = super().__call__(*args, **kwargs)

            for i, value in enumerate(args):
                para_name, para_type      = str_sig[i + 1]
                para_types, outer_type, _ = cls.get_para_type(type_string=para_type, sig_para=sig_para, para_name=para_name)

                value        = cls.string_transformation(string=value, para_types=para_types, outer_type=outer_type)
                setattr(obj, f"_{para_name}", value)

            for key, value in kwargs.items():
                para_type = list(filter(lambda input: input[0] == key, str_sig))

                para_types, outer_type, _ = cls.get_para_type(type_string=para_type[0][1], sig_para=sig_para, para_name=key)

                value = cls.string_transformation(string=value, para_types=para_types, outer_type=outer_type)

                if value is not None or obj.__dict__.get(f"_{key}") is None:
                    setattr(obj, f"_{key}", value)
        obj.main_process()
        return obj