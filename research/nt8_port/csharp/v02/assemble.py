#!/usr/bin/env python3.11
"""Single-source-of-truth assembler for the v0.2 ensemble core.

Inputs (edited by hand / generated):
  core_logic.cs.inc        -- the down-levelled decision-core logic (types only)
  _generated_data.cs.inc   -- frozen model + TMPL0 codebook constants (gen_data.py)

Outputs (never hand-edited):
  EnsembleCoreV02.region.cs           -- THE canonical shared region (usings + types)
  shim/EnsembleCoreV02.gen.cs         -- region wrapped in `namespace EnsembleV02Core {}`
  ../../../docs/nt8/7-EnsembleRunner_v0.2-RC.cs  -- region injected between its markers

The SAME region text lands in both the shim and the NinjaScript strategy, so the
core the shim proves at 100% parity is byte-identical to the code that ships in NT8.
verify_region.py re-checks that identity.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))

USINGS = "using System;\nusing System.Collections.Generic;\n"

BEGIN = "// ===SHARED-CORE-V02 BEGIN=== (single source: research/nt8_port/csharp/v02/EnsembleCoreV02.region.cs)"
END = "// ===SHARED-CORE-V02 END==="


def build_region():
    logic = open(os.path.join(HERE, "core_logic.cs.inc"), encoding="utf-8").read()
    data = open(os.path.join(HERE, "_generated_data.cs.inc"), encoding="utf-8").read()
    body = USINGS + "\n" + logic.rstrip("\n") + "\n\n" + data.rstrip("\n") + "\n"
    region = BEGIN + "\n" + body + END + "\n"
    return region


def main():
    region = build_region()
    # 1. canonical region file
    open(os.path.join(HERE, "EnsembleCoreV02.region.cs"), "w", newline="\n",
         encoding="utf-8").write(region)
    # 2. shim wrapper
    shim_dir = os.path.join(HERE, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    wrapped = "namespace EnsembleV02Core\n{\n" + region + "}\n"
    open(os.path.join(shim_dir, "EnsembleCoreV02.gen.cs"), "w", newline="\n",
         encoding="utf-8").write(wrapped)
    # 3. inject the region VERBATIM into the strategy between its BEGIN/END markers.
    #    C# is whitespace-insensitive, so the block sits at column 0 inside the
    #    nested namespace -> the bytes stay identical to the canonical region.
    strat = os.path.join(REPO, "docs", "nt8", "7-EnsembleRunner_v0.2-RC.cs")
    if os.path.exists(strat):
        txt = open(strat, encoding="utf-8").read()
        i0 = txt.find(BEGIN)
        i1 = txt.find(END)
        if i0 >= 0 and i1 > i0:
            new = txt[:i0] + region.rstrip("\n") + "\n" + txt[i1 + len(END):]
            open(strat, "w", newline="\n", encoding="utf-8").write(new)
            print("injected region into", strat)
        else:
            print("WARN: strategy markers not found; skipped injection")
    else:
        print("NOTE: strategy file not present yet; skipped injection")
    print("region bytes:", len(region))


if __name__ == "__main__":
    main()
