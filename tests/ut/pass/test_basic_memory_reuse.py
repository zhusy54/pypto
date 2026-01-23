# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto import ir
from pypto.ir import builder
from pypto.ir.op import block
from pypto.pypto_core import DataType, passes
from pypto.pypto_core import ir as core_ir


def get_var_memref_addr(func, var_name):
    """Extract MemRef address for a variable by name.

    Args:
        func: Function to search
        var_name: Name of the variable to find

    Returns:
        Address value if found, None otherwise
    """
    if not isinstance(func.body, ir.SeqStmts):
        return None

    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == var_name:
            if isinstance(stmt.var.type, core_ir.ShapedType):
                if stmt.var.type.memref is not None:
                    addr = stmt.var.type.memref.addr_
                    if isinstance(addr, core_ir.ConstInt):
                        return addr.value
    return None


def get_all_memref_addrs(func):
    """Get a dictionary of variable names to their MemRef addresses.

    Args:
        func: Function to analyze

    Returns:
        Dict[str, int]: Mapping of variable names to MemRef addresses
    """
    result = {}
    if not isinstance(func.body, ir.SeqStmts):
        return result

    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt):
            var_name = stmt.var.name
            if isinstance(stmt.var.type, core_ir.ShapedType):
                if stmt.var.type.memref is not None:
                    addr = stmt.var.type.memref.addr_
                    if isinstance(addr, core_ir.ConstInt):
                        result[var_name] = addr.value

    return result


def test_basic_memory_reuse_simple():
    """Test BasicMemoryReusePass with a simple buffer reuse case.

    核心概念：
    - let 赋值是引用语义，不涉及内存分配
    - 真正的内存复用是指：操作结果的底层缓冲区被后续操作复用
    - 例如：tile_e 的缓冲区可以复用 tile_a 的缓冲区（tile_a 生命周期已结束）
    """
    ib = builder.IRBuilder()

    with ib.function("test_basic_memory_reuse") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Constants for tile
        tile_height = 64
        tile_width = 64

        # Load tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # Compute: tile_c = tile_a + tile_b (tile_a is last used here)
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))

        # Compute: tile_d = tile_c * 2 (tile_b is last used in tile_c, tile_c is used here)
        tile_d = ib.let("tile_d", block.mul(tile_c, tile_c))

        # Compute: tile_e = tile_d + tile_d (tile_c is last used in tile_d)
        # tile_e can potentially reuse tile_a's memory since tile_a is no longer used
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))

        # Store result
        result = ib.let("result", block.store(tile_e, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass to allocate memory
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Get initial memory addresses (before reuse optimization)
    addrs_before = get_all_memref_addrs(func_with_memref)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Get memory addresses after optimization
    addrs_after = get_all_memref_addrs(optimized_func)

    # Verify the function is valid
    assert optimized_func is not None
    assert optimized_func.name == "test_basic_memory_reuse"
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Check that variables have memrefs
    stmts = optimized_func.body.stmts
    assert len(stmts) >= 5

    # All intermediate tiles should have memrefs
    for i in range(5):
        stmt = stmts[i]
        assert isinstance(stmt, ir.AssignStmt)
        var = stmt.var
        assert isinstance(var.type, core_ir.ShapedType)
        assert var.type.memref is not None

    # Verify memory reuse happened
    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 4] - defined at 3, last used in tile_e at 4
    # tile_e: [4, 4] - defined at 4, last used in store at 4
    #
    # Reuse opportunities:
    # - tile_d can reuse tile_a (lifetimes [3,4] and [0,2] don't overlap)
    # - tile_e can reuse tile_a or tile_b (lifetimes don't overlap)

    # Check that tile_d reuses tile_a's memory
    assert addrs_after["tile_d"] == addrs_after["tile_a"], \
        f"tile_d should reuse tile_a's memory: tile_d@{addrs_after.get('tile_d')} vs tile_a@{addrs_after.get('tile_a')}"

    # Check that tile_e reuses memory from tile_a (greedy first-fit)
    assert addrs_after["tile_e"] == addrs_after["tile_a"], \
        f"tile_e should reuse tile_a's memory: tile_e@{addrs_after.get('tile_e')} vs tile_a@{addrs_after.get('tile_a')}"

    # Verify that fewer unique addresses are used after optimization
    unique_addrs_after = len(set(addrs_after.values()))
    assert unique_addrs_after < len(addrs_after), \
        f"Expected memory reuse, but all {len(addrs_after)} variables have unique addresses"


def test_basic_memory_reuse_sequential():
    """Test BasicMemoryReusePass with sequential computation (ideal for buffer reuse).

    核心概念：
    - 顺序计算链中，每个操作的缓冲区在使用后立即失效
    - 后续操作的缓冲区可以复用前面已失效的缓冲区
    - 例如：tile_c 的缓冲区可以复用 tile_a 的缓冲区
    """
    ib = builder.IRBuilder()

    with ib.function("test_sequential_reuse") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Constants for tile
        tile_height = 64
        tile_width = 64

        # Load tile
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))

        # Sequential computations: each variable is only used once
        # tile_b = tile_a * 2
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))

        # tile_c = tile_b * 2 (tile_a is dead, tile_c can reuse tile_a's memory)
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))

        # tile_d = tile_c * 2 (tile_b is dead, tile_d can reuse tile_b's memory)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # tile_e = tile_d * 2 (tile_c is dead, tile_e can reuse tile_c's memory)
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))

        # Store result (tile_d is dead, only tile_e is live)
        result = ib.let("result", block.store(tile_e, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Get memory addresses after optimization
    addrs_after = get_all_memref_addrs(optimized_func)

    # Verify the function is valid
    assert optimized_func is not None
    assert optimized_func.name == "test_sequential_reuse"
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # All variables should have memrefs
    stmts = optimized_func.body.stmts
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            var = stmt.var
            if isinstance(var.type, core_ir.ShapedType):
                assert var.type.memref is not None

    # Expected lifetime analysis (sequential, no overlap):
    # tile_a: [0, 1] - defined at 0, last used in tile_b at 1
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 4] - defined at 3, last used in tile_e at 4
    # tile_e: [4, 4] - defined at 4, used in store at 4
    #
    # Optimal reuse strategy (greedy first-fit):
    # - tile_c reuses tile_a ([2,3] doesn't overlap [0,1])
    # - tile_d reuses tile_a ([3,4] doesn't overlap [0,1])
    # - tile_e reuses tile_a ([4,4] doesn't overlap [0,1])

    # Verify that tile_c, tile_d, and tile_e all reuse tile_a's memory
    assert addrs_after["tile_c"] == addrs_after["tile_a"], \
        f"tile_c should reuse tile_a's memory: tile_c@{addrs_after.get('tile_c')} vs tile_a@{addrs_after.get('tile_a')}"

    assert addrs_after["tile_d"] == addrs_after["tile_a"], \
        f"tile_d should reuse tile_a's memory: tile_d@{addrs_after.get('tile_d')} vs tile_a@{addrs_after.get('tile_a')}"

    assert addrs_after["tile_e"] == addrs_after["tile_a"], \
        f"tile_e should reuse tile_a's memory: tile_e@{addrs_after.get('tile_e')} vs tile_a@{addrs_after.get('tile_a')}"

    # Count unique memory addresses (should be minimal in sequential case)
    unique_addrs = len(set(addrs_after.values()))
    # We expect at most 2-3 unique addresses for this sequential pattern
    assert unique_addrs <= 3, \
        f"Sequential pattern should use at most 3 unique addresses, got {unique_addrs}"


def test_basic_memory_reuse_different_sizes():
    """Test BasicMemoryReusePass with different tensor sizes (size-aware buffer reuse).

    核心概念：
    - 大缓冲区可以被小缓冲区复用（有足够空间）
    - 小缓冲区不能被大缓冲区复用（空间不足）
    - 例如：64x64 的缓冲区可以复用另一个 64x64 的缓冲区，
           但 32x32 的缓冲区不能被 64x64 的缓冲区复用
    """
    ib = builder.IRBuilder()

    with ib.function("test_different_sizes") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([32, 32], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        # Load tiles of different sizes
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))  # 64x64 = 16384 bytes
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))  # 32x32 = 4096 bytes

        # Compute with tile_a
        tile_c = ib.let("tile_c", block.add(tile_a, tile_a))  # 64x64, tile_a dies

        # tile_d can potentially reuse tile_a's memory (both 64x64)
        # but NOT tile_b's memory (32x32 is too small)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))  # 64x64, tile_c dies

        # tile_e should be able to reuse the larger buffers (tile_a or tile_c)
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))  # 64x64, tile_d dies

        # Store result
        result = ib.let("result", block.store(tile_e, 0, 0, 64, 64, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Get memory addresses and sizes
    addrs_after = get_all_memref_addrs(optimized_func)

    # Verify the function is valid
    assert optimized_func is not None
    assert optimized_func.name == "test_different_sizes"

    # Verify all expected variables have memrefs
    for var_name in ["tile_a", "tile_b", "tile_c", "tile_d", "tile_e"]:
        assert var_name in addrs_after, f"{var_name} should have a memref"

    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 1] - defined at 1, never used (no dependencies)
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 4] - defined at 3, last used in tile_e at 4
    # tile_e: [4, 4] - defined at 4, used in store at 4
    #
    # Size-aware reuse opportunities:
    # - tile_d (16384 bytes) can reuse tile_a (16384 bytes) - same size ✓
    # - tile_e (16384 bytes) can reuse tile_a (16384 bytes) - same size ✓
    # - tile_d should NOT reuse tile_b (4096 bytes) - too small ✗
    # - tile_b (4096 bytes) could theoretically reuse tile_a (16384 bytes) if lifetimes allow

    # Verify that tile_d reuses tile_a's memory (same size, non-overlapping lifetimes)
    assert addrs_after["tile_d"] == addrs_after["tile_a"], \
        f"tile_d (64x64) should reuse tile_a (64x64) memory: tile_d@{addrs_after['tile_d']} vs tile_a@{addrs_after['tile_a']}"

    # Verify that tile_e reuses tile_a's memory
    assert addrs_after["tile_e"] == addrs_after["tile_a"], \
        f"tile_e (64x64) should reuse tile_a (64x64) memory: tile_e@{addrs_after['tile_e']} vs tile_a@{addrs_after['tile_a']}"

    # Extract MemRef objects to verify size checking
    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts
    vars_dict = {}
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            vars_dict[stmt.var.name] = stmt.var

    # Verify that variables have correct types
    tile_a_var = vars_dict.get("tile_a")
    tile_b_var = vars_dict.get("tile_b")
    tile_d_var = vars_dict.get("tile_d")

    # Basic type checks
    if tile_a_var:
        assert isinstance(tile_a_var.type, core_ir.ShapedType), "tile_a should be ShapedType"
    if tile_b_var:
        assert isinstance(tile_b_var.type, core_ir.ShapedType), "tile_b should be ShapedType"
    if tile_d_var:
        assert isinstance(tile_d_var.type, core_ir.ShapedType), "tile_d should be ShapedType"


def test_basic_memory_reuse_empty_function():
    """Test BasicMemoryReusePass with empty function (edge case)."""
    ib = builder.IRBuilder()

    with ib.function("test_empty") as f:
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))
        ib.return_stmt(output)

    func = f.get_result()

    # Run BasicMemoryReusePass on empty function
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func)

    # Should return a valid function (even if unchanged)
    assert optimized_func is not None
    assert optimized_func.name == "test_empty"


def test_basic_memory_reuse_memref_sharing():
    """Test that MemRef objects are actually shared between variables."""
    ib = builder.IRBuilder()

    with ib.function("test_memref_sharing") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Create a chain of operations
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))  # tile_a dies
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))  # tile_b dies, can reuse tile_a
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))  # tile_c dies, can reuse tile_b
        result = ib.let("result", block.store(tile_d, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Extract variables from optimized function
    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts

    # Extract variables
    vars_dict = {}
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            vars_dict[stmt.var.name] = stmt.var

    # Get memory addresses
    addrs = get_all_memref_addrs(optimized_func)

    # Expected lifetime analysis:
    # tile_a: [0, 1] - defined at 0, last used in tile_b at 1
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 3] - defined at 3, used in store at 3
    #
    # Reuse opportunities:
    # - tile_c can reuse tile_a ([2,3] doesn't overlap [0,1])
    # - tile_d can reuse tile_a ([3,3] doesn't overlap [0,1])

    # Check that tile_c should reuse tile_a's MemRef (both are size 16384, non-overlapping lifetimes)
    tile_a_var = vars_dict.get("tile_a")
    tile_c_var = vars_dict.get("tile_c")

    assert tile_a_var is not None, "tile_a not found in optimized function"
    assert tile_c_var is not None, "tile_c not found in optimized function"

    tile_a_type = tile_a_var.type
    tile_c_type = tile_c_var.type

    assert isinstance(tile_a_type, core_ir.ShapedType), "tile_a should be ShapedType"
    assert isinstance(tile_c_type, core_ir.ShapedType), "tile_c should be ShapedType"

    assert tile_a_type.memref is not None, "tile_a should have MemRef"
    assert tile_c_type.memref is not None, "tile_c should have MemRef"

    # Verify memory space is the same
    assert tile_a_type.memref.memory_space_ == tile_c_type.memref.memory_space_, \
        "tile_a and tile_c should be in same memory space"

    # Verify size is the same
    assert tile_a_type.memref.size_ == tile_c_type.memref.size_, \
        "tile_a and tile_c should have same size"

    # Verify addresses are the same (actual reuse check)
    assert addrs["tile_c"] == addrs["tile_a"], \
        f"tile_c should reuse tile_a's memory address: tile_c@{addrs['tile_c']} vs tile_a@{addrs['tile_a']}"

    # Also check tile_d reuses tile_a
    assert addrs["tile_d"] == addrs["tile_a"], \
        f"tile_d should reuse tile_a's memory address: tile_d@{addrs['tile_d']} vs tile_a@{addrs['tile_a']}"


def test_basic_memory_reuse_with_dependencies():
    """Test BasicMemoryReusePass respects dependencies."""
    ib = builder.IRBuilder()

    with ib.function("test_dependencies") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Load two tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # tile_c depends on both tile_a and tile_b
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))

        # tile_d depends on tile_c (tile_a and tile_b can be reused after this)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # tile_e depends on tile_d (tile_c can be reused after this)
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))

        # Store result
        result = ib.let("result", block.store(tile_e, 0, 0, tile_height, tile_width, output))

        ib.return_stmt(result)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Get memory addresses
    addrs = get_all_memref_addrs(optimized_func)

    # Should not crash and produce valid output
    assert optimized_func is not None
    assert isinstance(optimized_func.body, ir.SeqStmts)

    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 3] - defined at 2, last used in tile_d at 3
    # tile_d: [3, 4] - defined at 3, last used in tile_e at 4
    # tile_e: [4, 4] - defined at 4, used in store at 4
    #
    # Reuse opportunities (respecting dependencies):
    # - tile_d can reuse tile_a or tile_b (lifetimes [3,4] vs [0,2] and [1,2])
    # - tile_e can reuse tile_a or tile_b (lifetimes [4,4] vs [0,2] and [1,2])
    # - tile_e can also reuse tile_c (lifetimes [4,4] vs [2,3])

    # Verify all variables have memrefs
    for var_name in ["tile_a", "tile_b", "tile_c", "tile_d", "tile_e"]:
        assert var_name in addrs, f"{var_name} should have a memref"

    # Verify memory reuse happened
    # tile_d should reuse tile_a's memory (greedy first-fit)
    assert addrs["tile_d"] == addrs["tile_a"], \
        f"tile_d should reuse tile_a's memory: tile_d@{addrs['tile_d']} vs tile_a@{addrs['tile_a']}"

    # tile_e should also reuse tile_a's memory
    assert addrs["tile_e"] == addrs["tile_a"], \
        f"tile_e should reuse tile_a's memory: tile_e@{addrs['tile_e']} vs tile_a@{addrs['tile_a']}"

    # Verify that fewer unique addresses are used after optimization
    unique_addrs = len(set(addrs.values()))
    assert unique_addrs < len(addrs), \
        f"Expected memory reuse, but all {len(addrs)} variables have unique addresses"


def test_basic_memory_reuse_multiple_memory_spaces():
    """Test BasicMemoryReusePass doesn't mix different memory spaces."""
    ib = builder.IRBuilder()

    with ib.function("test_memory_spaces") as f:
        # This test verifies that variables in DDR don't reuse UB memory and vice versa
        # Parameters are in DDR, tiles are in UB
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output_a = f.param("output_a", ir.TensorType([64, 64], DataType.FP32))
        output_b = f.param("output_b", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_height = 64
        tile_width = 64

        # Load creates UB tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

        # Compute creates more UB tiles
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))  # tile_a and tile_b die here

        # Store to first output (intermediate result)
        result_a = ib.let("result_a", block.store(tile_c, 0, 0, tile_height, tile_width, output_a))

        # More UB computation (tile_c dies here)
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))

        # Store final result
        result_b = ib.let("result_b", block.store(tile_d, 0, 0, tile_height, tile_width, output_b))

        ib.return_stmt(result_b)

    func = f.get_result()

    # First run InitMemRefPass
    init_pass = passes.InitMemRefPass()
    func_with_memref = init_pass.run(func)

    # Then run BasicMemoryReusePass
    reuse_pass = passes.BasicMemoryReusePass()
    optimized_func = reuse_pass.run(func_with_memref)

    # Verify function structure
    assert isinstance(optimized_func.body, ir.SeqStmts)
    stmts = optimized_func.body.stmts

    # Verify DDR parameters don't get mixed with UB variables
    params = {p.name: p for p in optimized_func.params}
    for param_name in ["input_a", "input_b", "output_a", "output_b"]:
        p = params[param_name]
        assert isinstance(p.type, core_ir.ShapedType)
        if p.type.memref is not None:
            assert p.type.memref.memory_space_ == core_ir.MemorySpace.DDR, \
                f"{param_name} should be in DDR memory space"

    # Verify all UB variables stay in UB
    ub_vars_found = 0
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt):
            var = stmt.var
            if var.name in ["tile_a", "tile_b", "tile_c", "tile_d"]:
                assert isinstance(var.type, core_ir.ShapedType)
                if var.type.memref is not None:
                    assert var.type.memref.memory_space_ == core_ir.MemorySpace.UB, \
                        f"{var.name} should be in UB memory space"
                    ub_vars_found += 1

    # Ensure we found all expected UB variables
    assert ub_vars_found == 4, f"Expected 4 UB variables, found {ub_vars_found}"

    # Get memory addresses for UB variables
    addrs = get_all_memref_addrs(optimized_func)

    # Expected lifetime analysis:
    # tile_a: [0, 2] - defined at 0, last used in tile_c at 2
    # tile_b: [1, 2] - defined at 1, last used in tile_c at 2
    # tile_c: [2, 4] - defined at 2, used in result_a at 3 and tile_d at 4
    # tile_d: [4, 4] - defined at 4, used in result_b at 4
    #
    # All are in UB memory space, reuse should happen within UB:
    # - tile_d should reuse tile_a or tile_b's memory (non-overlapping lifetimes)

    # Verify that tile_d reuses UB memory from tile_a
    assert addrs["tile_d"] == addrs["tile_a"], \
        f"tile_d should reuse tile_a's UB memory: tile_d@{addrs['tile_d']} vs tile_a@{addrs['tile_a']}"
