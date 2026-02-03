# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compilation tests for block.loadex operation."""

import os

import pytest
from pypto import ir
from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.ir.op.block_ops import LayoutOpType
from pypto.pypto_core import DataType


class TestLoadexCompilation:
    """Test block.loadex operation with full compilation pipeline."""

    def test_loadex_compile_with_cce(self):
        """Test loadex operation with CCE backend compilation."""
        # Build a function using loadex
        ib = IRBuilder()

        with ib.function("loadex_compile_test") as f:
            # Define input tensor [16, 32]
            input_tensor = f.param("input", ir.TensorType([16, 32], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([32, 4], DataType.FP32))
            f.return_type(ir.TensorType([32, 4], DataType.FP32))

            # Apply loadex with transformations:
            # 1. VIEW: Load [16, 32] region from (0, 0)
            # 2. TRANSPOSE: [16, 32] -> [32, 16]
            # 3. VIEW: [32, 16] -> [32, 4] with offset [0, 0]
            ops = [
                (LayoutOpType.VIEW, [16, 32], [0, 0]),   # Load region
                (LayoutOpType.TRANSPOSE, 0, 1),           # [16, 32] -> [32, 16]
                (LayoutOpType.VIEW, [32, 4], [0, 0]),     # [32, 16] -> [32, 4]
            ]
            tile_transformed = ib.let("tile_transformed", block.loadex(input_tensor, ops))

            # Store result back to output tensor
            result = ib.let("result", block.store(tile_transformed, 0, 0, 32, 4, output_tensor))

            # Return result
            ib.return_stmt(result)

        # Get the function
        func = f.get_result()

        # Create a program
        span = ir.Span.unknown()
        program = ir.Program([func], "loadex_test_program", span)

        # Compile with CCE backend
        output_path = ir.compile(
            program,
            strategy=ir.OptimizationStrategy.Default,
            dump_passes=False,  # Disable pass dumping for testing
            codegen=ir.CodegenBackend.CCE,
        )

        # Verify output directory exists
        assert os.path.exists(output_path)
        assert os.path.isdir(output_path)

        # Verify that some files were generated
        # CCE backend should generate .cpp files
        generated_files = os.listdir(output_path)
        assert len(generated_files) > 0

        # Print generated files for debugging
        print(f"Generated files in {output_path}:")
        for fname in generated_files:
            print(f"  - {fname}")

    def test_loadex_compile_with_pto(self):
        """Test loadex operation with PTO backend compilation."""
        # Build a function using loadex
        ib = IRBuilder()

        with ib.function("loadex_pto_test") as f:
            # Define input tensor [8, 16]
            input_tensor = f.param("input", ir.TensorType([8, 16], DataType.FP16))
            output_tensor = f.param("output", ir.TensorType([32, 4], DataType.FP16))
            f.return_type(ir.TensorType([32, 4], DataType.FP16))

            # Apply loadex with transformations:
            # 1. VIEW: Load [8, 16] region from (0, 0)
            # 2. TRANSPOSE: [8, 16] -> [16, 8]
            # 3. RESHAPE: [16, 8] -> [32, 4]
            ops = [
                (LayoutOpType.VIEW, [8, 16], [0, 0]),
                (LayoutOpType.TRANSPOSE, 0, 1),
                (LayoutOpType.RESHAPE, [32, 4]),
            ]
            tile_transformed = ib.let("tile_transformed", block.loadex(input_tensor, ops))

            # Store result
            result = ib.let("result", block.store(tile_transformed, 0, 0, 32, 4, output_tensor))

            # Return result
            ib.return_stmt(result)

        # Get the function and create program
        func = f.get_result()
        span = ir.Span.unknown()
        program = ir.Program([func], "loadex_pto_program", span)

        # Compile with PTO backend
        output_path = ir.compile(
            program,
            strategy=ir.OptimizationStrategy.Default,
            dump_passes=False,
            codegen=ir.CodegenBackend.PTO,
        )

        # Verify output directory exists
        assert os.path.exists(output_path)
        assert os.path.isdir(output_path)

        # Verify PTO file was generated
        pto_file = os.path.join(output_path, "output.pto")
        assert os.path.exists(pto_file)

        # Read and verify PTO content is not empty
        with open(pto_file) as f:
            pto_content = f.read()
            assert len(pto_content) > 0
            print(f"Generated PTO file size: {len(pto_content)} bytes")

    def test_loadex_multiple_ops_compile(self):
        """Test loadex with three operations and CCE compilation."""
        ib = IRBuilder()

        with ib.function("loadex_multi_ops") as f:
            # Input: [16, 32] FP32
            input_tensor = f.param("input", ir.TensorType([16, 32], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([64, 2], DataType.FP32))
            f.return_type(ir.TensorType([64, 2], DataType.FP32))

            # Apply four transformations:
            # 1. VIEW: Load [16, 32] region from (0, 0)
            # 2. TRANSPOSE: [16, 32] -> [32, 16]
            # 3. RESHAPE: [32, 16] -> [64, 8]
            # 4. VIEW: [64, 8] -> [64, 2] with offset [0, 0]
            ops = [
                (LayoutOpType.VIEW, [16, 32], [0, 0]),
                (LayoutOpType.TRANSPOSE, 0, 1),
                (LayoutOpType.RESHAPE, [64, 8]),
                (LayoutOpType.VIEW, [64, 2], [0, 0]),
            ]
            tile_transformed = ib.let("tile_transformed", block.loadex(input_tensor, ops))

            # Store result
            result = ib.let("result", block.store(tile_transformed, 0, 0, 64, 2, output_tensor))

            ib.return_stmt(result)

        # Create program and compile
        func = f.get_result()
        program = ir.Program([func], "loadex_multi_ops", ir.Span.unknown())

        output_path = ir.compile(program, strategy=ir.OptimizationStrategy.Default, dump_passes=False)

        assert os.path.exists(output_path)
        # Verify some files were generated
        files = os.listdir(output_path)
        assert len(files) > 0
        print(f"Multi-ops test generated {len(files)} files")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
