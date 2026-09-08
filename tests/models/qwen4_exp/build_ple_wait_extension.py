# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build FreeToken's existing PLE extension with the serving Python/torch ABI.

Run inside the candidate serving environment, with a read-only FreeToken source
checkout mounted. This builds only ple_store_ext.cpp, without importing or
installing the FreeToken package and its unrelated kernels. Add the output
directory to PYTHONPATH when starting the server. Do not copy a binary from a
different Python/torch environment.
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freetoken-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = (
        args.freetoken_source.resolve()
        / "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp"
    )
    if not source.is_file():
        parser.error(f"FreeToken PLE source not found: {source}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load(
        name="vllm_ple_wait_ext",
        sources=[str(source)],
        build_directory=str(output),
        extra_cflags=["-O3", "-std=c++17", "-pthread"],
        extra_ldflags=["-pthread", "-ldl"],
        with_cuda=False,
        verbose=True,
    )
    for name in ("memop_wait_reset", "memop_write", "memop_wait_geq", "signal_flag"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"PLE extension is missing {name}")
    print(f"Built {module.__file__}; expose {output} through PYTHONPATH")
    print("Build/import only: CUDA stream wait and graph capture remain untested.")


if __name__ == "__main__":
    main()
