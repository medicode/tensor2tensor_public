import traceback
import types
import sys
from pprint import pprint
from typing import Set

T2T_PREFIXES = {
    "tensor2tensor.",
}
DT_PREFIXES = {
    "fh_platform",
    "fathomt2t",
    "fathomt2t_dependencies",
    "pretrained_models",
    "fathomtf",
    # "fathomairflow"
}
ALL_PREFIXES = T2T_PREFIXES.union(DT_PREFIXES)

TOTAL_NUM_IMPORTS = 339
TOTAL_T2T_IMPORTS = 175
TOTAL_DT_IMPORTS = TOTAL_NUM_IMPORTS - TOTAL_T2T_IMPORTS


def global_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__


def module_imports(prefixes: Set[str] = set()):
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
        print("Failure!")
        traceback.print_exc()
    else:
        print("Success!")
    finally:
        imports = sorted(set(module_imports(prefixes)))
        pct = len(imports) / total_num_modules * 100
        print(f"Imported {len(imports)}/{total_num_modules} ({pct}%) modules")
        # pprint(imports)


test_import(
    command="from tensor2tensor.bin import t2t_trainer",
    prefixes=T2T_PREFIXES,
    total_num_modules=TOTAL_T2T_IMPORTS
)

test_import(
    command="from tensor2tensor.bin import t2t_trainer",
    prefixes=DT_PREFIXES,
    total_num_modules=TOTAL_DT_IMPORTS
)

test_import(
    command="from tensor2tensor.bin import t2t_trainer",
    prefixes=ALL_PREFIXES,
    total_num_modules=TOTAL_NUM_IMPORTS
)