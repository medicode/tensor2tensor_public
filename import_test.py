import traceback
import types
import sys
from typing import Set

T2T_PREFIX = "tensor2tensor."
FH_T2T = "fathomt2t"
FH_T2T_DEPS = "fathomt2t_dependencies"

TOTAL_NUM_IMPORTS = 106


def global_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__


def module_imports(prefixes: Set[str]=set()):
    return [
        key for key in
        sys.modules.keys()
        if not prefixes or any(
            key.startswith(prefix) for prefix in prefixes
        )
    ]


def test_import(command: str, prefixes: Set[str], total_num_modules: int):
    try:
        exec(command)
    except Exception:
        imports = sorted(set(module_imports(prefixes)))
        pct = len(imports) / total_num_modules * 100
        traceback.print_exc()
        print(f"Imported {len(imports)}/{total_num_modules} ({pct}%) modules")
        print(imports)
    else:
        print("Success!")


test_import(
    command="from tensor2tensor.bin import t2t_trainer",
    prefixes={T2T_PREFIX, FH_T2T, FH_T2T_DEPS},
    total_num_modules=TOTAL_NUM_IMPORTS
)



def find_all_imports