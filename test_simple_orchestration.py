#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Simple Orchestration Codegen Test

Tests basic orchestration code generation with a simple linear task graph:
    c = add(a, b)
    d = mul(c, a)
"""

from pypto import DataType, ir
from pypto.ir import IRBuilder


def build_simple_orchestration_ir():
    """Build a simple orchestration function for testing.

    Formula: d = (a + b) * a
    """
    ib = IRBuilder()

    with ib.function("simple_orch") as f:
        # Parameters
        a = f.param("a", ir.TensorType([128], DataType.FP32))
        b = f.param("b", ir.TensorType([128], DataType.FP32))
        d = f.param("d", ir.TensorType([128], DataType.FP32))
        f.return_type(ir.TensorType([128], DataType.FP32))

        # Task 1: c = add(a, b)
        add_op = ir.GlobalVar("kernel_add")
        # Use kwargs dict to pass metadata
        kwargs = {"func_id": 0, "device_type": 1}  # AIV
        add_call = ir.Call(add_op, [a, b], kwargs, ir.Span.unknown())
        c = ib.let("c", add_call)

        # Task 2: d = mul(c, a)
        mul_op = ir.GlobalVar("kernel_mul")
        kwargs = {"func_id": 1, "device_type": 1}  # AIV
        mul_call = ir.Call(mul_op, [c, a], kwargs, ir.Span.unknown())
        result = ib.let("result", mul_call)

        # Return output
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "simple_orch", ir.Span.unknown())
    return program


def main():
    print("=" * 70)
    print("Simple Orchestration Codegen Test")
    print("=" * 70)

    # Step 1: Build IR
    print("\n[1] Building simple orchestration IR...")
    program = build_simple_orchestration_ir()
    print("✓ IR construction complete")

    # Step 2: Print IR (preview)
    print("\n[2] IR Preview:")
    print("-" * 70)
    ir_text = ir.python_print(program)
    print(ir_text[:500])
    print("..." if len(ir_text) > 500 else "")
    print("-" * 70)

    # Step 3: Generate code using PTOCodegen
    print("\n[3] Generating code using PTOCodegen...")
    print("    (PTOCodegen now generates both kernel PTO assembly and orchestration C++ code)")
    codegen = ir.PTOCodegen()
    generated_code = codegen.generate(program)
    print("✓ Code generation complete")

    # Step 4: Display generated code
    print("\n[4] Generated Code:")
    print("=" * 70)
    print(generated_code)
    print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\nGenerated function:")
    print("  - Function name: simple_orch")
    print("  - Function type: Orchestration (detected by TensorType parameters)")
    print("  - Parameters: a, b, d")
    print("  - Tasks: 2 (kernel_add, kernel_mul)")
    print("\nThe generated code shows:")
    print("  - Function classification (Orchestration vs Kernel)")
    print("  - Placeholder for orchestration C++ code generation")
    print("  - Next step: Implement full C++ code generation")
    print("=" * 70)


if __name__ == "__main__":
    main()
