# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Block operations for PyPTO IR.

Block operations work on TileType (unified buffer) and support block-level programming.
These operations include memory operations (load, store), element-wise operations,
unary operations, and reduction operations.
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, Span

from ..utils import _get_span_or_capture, _normalize_expr


class LayoutOpType(IntEnum):
    """Layout transformation operation types for block.loadex.

    Enum values must match C++ LayoutOpTypeEnum in transform.cpp.
    """

    VIEW = 0
    RESHAPE = 1
    TRANSPOSE = 2


# ============================================================================
# Memory Operations
# ============================================================================


def load(
    tensor: Expr,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    target_memory: int = 1,
    span: Optional[Span] = None,
) -> Call:
    """Copy data from tensor to specified memory level.

    Args:
        tensor: Source tensor (TensorType)
        row_offset: Row offset in the tensor (scalar)
        col_offset: Column offset in the tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)
        target_memory: Target memory space for the output tile.
                     1=UB (UB, default), 2=L1.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with the copied data
    """
    # Validate target_memory: only UB(1) and L1(2) are allowed for load
    if target_memory not in (1, 2):
        raise ValueError(f"target_memory for block.load must be 1 (UB) or 2 (L1), got {target_memory}")

    actual_span = _get_span_or_capture(span)
    args = [
        tensor,
        _normalize_expr(row_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(height, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(width, actual_span, int_dtype=DataType.INT32),
    ]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {"target_memory": target_memory}

    return _ir_core.create_op_call("block.load", args, kwargs, actual_span)


def store(
    tile: Expr,
    row_offset: Union[int, Expr],
    col_offset: Union[int, Expr],
    height: Union[int, Expr],
    width: Union[int, Expr],
    output_tensor: Expr,
    span: Optional[Span] = None,
) -> Call:
    """Copy data from unified buffer (tile) to tensor.

    Args:
        tile: Source tile (TileType)
        row_offset: Row offset in the output tensor (scalar)
        col_offset: Column offset in the output tensor (scalar)
        height: Height of the tile to copy (scalar)
        width: Width of the tile to copy (scalar)
        output_tensor: Output tensor (TensorType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns the output tensor
    """
    actual_span = _get_span_or_capture(span)
    args = [
        tile,
        _normalize_expr(row_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(col_offset, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(height, actual_span, int_dtype=DataType.INT32),
        _normalize_expr(width, actual_span, int_dtype=DataType.INT32),
        output_tensor,
    ]
    return _ir_core.create_op_call("block.store", args, {}, actual_span)


def move(
    tile: Expr,
    target_memory: int,
    transpose: bool = False,
    span: Optional[Span] = None,
) -> Call:
    """Move tile between memory levels with optional transpose.

    Args:
        tile: Input tile (TileType)
        target_memory: Target memory space (1=UB, 2=L1, 3=L0A, 4=L0B)
        transpose: Whether to transpose the tile (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType in the target memory space
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {
        "target_memory": target_memory,
        "transpose": transpose,
    }

    return _ir_core.create_op_call("block.move", args, kwargs, actual_span)


# ============================================================================
# Element-wise Operations
# ============================================================================


def mul(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise multiplication of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.mul", [lhs, rhs], {}, actual_span)


def add(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise addition of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.add", [lhs, rhs], {}, actual_span)


def div(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise division of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.div", [lhs, rhs], {}, actual_span)


def sub(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise subtraction of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.sub", [lhs, rhs], {}, actual_span)


def muls(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise multiplication of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.muls", [lhs, rhs_expr], {}, actual_span)


def adds(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise addition of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.adds", [lhs, rhs_expr], {}, actual_span)


def divs(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise division of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.divs", [lhs, rhs_expr], {}, actual_span)


def subs(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise subtraction of tile and scalar.

    Args:
        lhs: Tile (TileType)
        rhs: Scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("block.subs", [lhs, rhs_expr], {}, actual_span)


# ============================================================================
# Unary Operations
# ============================================================================


def neg(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise negation of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise negation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.neg", [tile], {}, actual_span)


def exp(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise exponential function of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise exponential
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.exp", [tile], {}, actual_span)


def recip(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise reciprocal (1/x) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.recip", [tile], {}, actual_span)


def sqrt(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise square root of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.sqrt", [tile], {}, actual_span)


def rsqrt(tile: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise reciprocal square root (1/sqrt(x)) of a tile.

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.rsqrt", [tile], {}, actual_span)


# ============================================================================
# Matrix Operations
# ============================================================================


def matmul(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Matrix multiplication of two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.matmul", [lhs, rhs], {}, actual_span)


def matmul_acc(acc: Expr, lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Matrix multiplication with accumulation.

    Performs matrix multiplication and accumulates the result: acc = acc + lhs @ rhs.
    This is commonly used in loop-based matrix multiplication where results are
    accumulated over the K dimension.

    Args:
        acc: Accumulator tile (TileType) to accumulate into
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication with accumulation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.matmul_acc", [acc, lhs, rhs], {}, actual_span)


# ============================================================================
# Reduction Operations
# ============================================================================


def max(tile: Expr, axis: int, keepdim: bool = False, span: Optional[Span] = None) -> Call:
    """Max reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for max reduction
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.max", args, kwargs, actual_span)


def row_max(tile: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise max reduction of a tile.

    This is a convenience function equivalent to max(tile, axis=1, keepdim=True).
    Output shape is [rows, 1].

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise max reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_max", [tile], {}, actual_span)


def row_sum(tile: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise sum reduction of a tile.

    This is a convenience function equivalent to sum(tile, axis=1, keepdim=True).
    Output shape is [rows, 1].

    Args:
        tile: Input tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise sum reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_sum", [tile], {}, actual_span)


# ============================================================================
# Row Broadcast Operations
# ============================================================================


def row_expand_sub(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise broadcast subtraction.

    Subtracts a row vector from each row of the tile.
    tile[i, :] - row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_expand_sub", [tile, row_vec], {}, actual_span)


def row_expand_div(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise broadcast division.

    Divides each row of the tile by the corresponding row vector value.
    tile[i, :] / row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_expand_div", [tile, row_vec], {}, actual_span)


def row_expand_mul(tile: Expr, row_vec: Expr, span: Optional[Span] = None) -> Call:
    """Row-wise broadcast multiplication.

    Multiplies each row of the tile by the corresponding row vector value.
    tile[i, :] * row_vec[i, 0] for all i.

    Args:
        tile: Input tile (TileType [M, N])
        row_vec: Row vector (TileType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.row_expand_mul", [tile, row_vec], {}, actual_span)


def maximum(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise maximum of two tiles.

    Supports broadcasting for two tiles.

    Args:
        lhs: Left-hand side tile (TileType)
        rhs: Right-hand side tile (TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("block.maximum", [lhs, rhs], {}, actual_span)


# ============================================================================
# Reduction Operations (continued)
# ============================================================================


def sum(tile: Expr, axis: int, keepdim: bool = False, span: Optional[Span] = None) -> Call:
    """Sum reduction of a tile along specified axis.

    Args:
        tile: Input tile (TileType)
        axis: Reduction axis (0 for row reduction, 1 for column reduction, -1 for last axis)
        keepdim: Whether to keep the reduced dimension as 1 (default: False)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for sum reduction
    """

    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {
        "axis": axis,
        "keepdim": keepdim,
    }

    return _ir_core.create_op_call("block.sum", args, kwargs, actual_span)


# ============================================================================
# Transform Operations
# ============================================================================


def view(
    tile: Expr, shape: List[Union[int, Expr]], offset: List[Union[int, Expr]], span: Optional[Span] = None
) -> Call:
    """Create a view/slice of a tile with new shape and offset.

    Args:
        tile: Input tile expression
        shape: New shape dimensions (at most 2 for TileType)
        offset: Offset dimensions for the view
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tile view
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Add the number of shape dimensions as a ConstInt
    # This allows the C++ side to correctly split shape and offset arguments
    args.append(ConstInt(len(shape), DataType.INT32, actual_span))

    # Add shape dimensions
    for dim in shape:
        args.append(_normalize_expr(dim, actual_span, int_dtype=DataType.INT32))

    # Add offset dimensions
    for off in offset:
        args.append(_normalize_expr(off, actual_span, int_dtype=DataType.INT32))

    return _ir_core.create_op_call("block.view", args, {}, actual_span)


def reshape(tile: Expr, shape: List[Union[int, Expr]], span: Optional[Span] = None) -> Call:
    """Reshape tile to new shape.

    Args:
        tile: Input tile expression
        shape: New shape dimensions (at most 2 for TileType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile reshape
    """
    actual_span = _get_span_or_capture(span)
    args = [tile]

    # Add the number of shape dimensions as a ConstInt
    # This allows the C++ side to correctly parse shape arguments
    args.append(ConstInt(len(shape), DataType.INT32, actual_span))

    # Add shape dimensions
    for dim in shape:
        args.append(_normalize_expr(dim, actual_span, int_dtype=DataType.INT32))

    return _ir_core.create_op_call("block.reshape", args, {}, actual_span)


def transpose(tile: Expr, axis1: int, axis2: int, span: Optional[Span] = None) -> Call:
    """Transpose tile by swapping two axes.

    Args:
        tile: Input tile expression
        axis1: First axis to swap (supports negative indexing)
        axis2: Second axis to swap (supports negative indexing)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tile transpose
    """
    actual_span = _get_span_or_capture(span)

    # Create ConstInt for axis indices
    axis1_expr = ConstInt(axis1, DataType.INT32, actual_span)
    axis2_expr = ConstInt(axis2, DataType.INT32, actual_span)

    args = [tile, axis1_expr, axis2_expr]

    return _ir_core.create_op_call("block.transpose", args, {}, actual_span)


def loadex(
    tensor: Expr,
    ops: List[tuple],
    target_memory: int = 1,
    span: Optional[Span] = None,
) -> Call:
    """Load data from tensor with layout transformations applied.

    This operation combines loading from tensor memory with layout transformations
    (view, reshape, transpose) into a single optimized operation. The first operation
    in the ops list determines what region to load from the tensor.

    Args:
        tensor: Source tensor (TensorType)
        ops: List of layout operations. Each operation is a tuple:
             - VIEW: (LayoutOpType.VIEW, [h, w], [off_h, off_w])
                    When used as first op: specifies region to load from tensor
                    When used later: specifies slice within tile
             - RESHAPE: (LayoutOpType.RESHAPE, [new_h, new_w])
             - TRANSPOSE: (LayoutOpType.TRANSPOSE, axis1, axis2)

             If first op is not VIEW, the entire tensor will be loaded.
        target_memory: Target memory space for the output tile.
                     1=UB (default), 2=L1.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TileType with transformed data

    Raises:
        ValueError: If ops list is empty, exceeds 8 operations, or contains invalid operation types

    Example:
        from pypto.ir.op.block_ops import LayoutOpType

        # Load [16, 32] region from (0, 0) and apply transpose
        ops = [
            (LayoutOpType.VIEW, [16, 32], [0, 0]),  # Load region
            (LayoutOpType.TRANSPOSE, 0, 1),         # [16, 32] -> [32, 16]
        ]
        tile = block.loadex(input_tensor, ops)

        # Load entire tensor and transpose
        ops = [(LayoutOpType.TRANSPOSE, 0, 1)]
        tile = block.loadex(input_tensor, ops)
    """
    # Validate target_memory
    if target_memory not in (1, 2):
        raise ValueError(f"target_memory for block.loadex must be 1 (UB) or 2 (L1), got {target_memory}")

    actual_span = _get_span_or_capture(span)

    # Validate ops list
    if len(ops) == 0:
        raise ValueError("block.loadex requires at least one operation")
    if len(ops) > 8:
        raise ValueError(f"block.loadex supports at most 8 operations, got {len(ops)}")

    # Build args: tensor, num_ops
    args = [tensor]
    args.append(ConstInt(len(ops), DataType.INT32, actual_span))

    # Encode operations into flattened array
    encoded_data = []
    for i, op in enumerate(ops):
        if not isinstance(op, tuple) or len(op) < 2:
            raise ValueError(f"Operation {i} must be a tuple with at least 2 elements (op_type, ...)")

        op_type = op[0]
        if not isinstance(op_type, LayoutOpType):
            raise ValueError(f"Operation {i} type must be LayoutOpType, got {type(op_type)}")

        encoded_data.append(ConstInt(int(op_type), DataType.INT32, actual_span))

        if op_type == LayoutOpType.VIEW:
            # VIEW: (op_type, shape_list, offset_list)
            if len(op) != 3:
                raise ValueError(
                    f"VIEW operation {i} requires 3 elements (op_type, shape, offset), got {len(op)}"
                )

            shape = op[1]
            offset = op[2]

            if not isinstance(shape, list) or not isinstance(offset, list):
                raise ValueError(f"VIEW operation {i}: shape and offset must be lists")

            if len(shape) != len(offset):
                raise ValueError(
                    f"VIEW operation {i}: shape and offset must have same length, got {len(shape)} and {len(offset)}"
                )

            param_count = len(shape) + len(offset)
            encoded_data.append(ConstInt(param_count, DataType.INT32, actual_span))

            # Encode shape dimensions
            for dim in shape:
                encoded_data.append(_normalize_expr(dim, actual_span, int_dtype=DataType.INT32))

            # Encode offset dimensions
            for off in offset:
                encoded_data.append(_normalize_expr(off, actual_span, int_dtype=DataType.INT32))

        elif op_type == LayoutOpType.RESHAPE:
            # RESHAPE: (op_type, shape_list)
            if len(op) != 2:
                raise ValueError(f"RESHAPE operation {i} requires 2 elements (op_type, shape), got {len(op)}")

            shape = op[1]

            if not isinstance(shape, list):
                raise ValueError(f"RESHAPE operation {i}: shape must be a list")

            param_count = len(shape)
            encoded_data.append(ConstInt(param_count, DataType.INT32, actual_span))

            # Encode shape dimensions
            for dim in shape:
                encoded_data.append(_normalize_expr(dim, actual_span, int_dtype=DataType.INT32))

        elif op_type == LayoutOpType.TRANSPOSE:
            # TRANSPOSE: (op_type, axis1, axis2)
            if len(op) != 3:
                raise ValueError(
                    f"TRANSPOSE operation {i} requires 3 elements (op_type, axis1, axis2), got {len(op)}"
                )

            axis1 = op[1]
            axis2 = op[2]

            if not isinstance(axis1, int) or not isinstance(axis2, int):
                raise ValueError(f"TRANSPOSE operation {i}: axis1 and axis2 must be integers")

            param_count = 2
            encoded_data.append(ConstInt(param_count, DataType.INT32, actual_span))
            encoded_data.append(ConstInt(axis1, DataType.INT32, actual_span))
            encoded_data.append(ConstInt(axis2, DataType.INT32, actual_span))

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    # Add encoding length
    encoding_length = len(encoded_data)
    args.append(ConstInt(encoding_length, DataType.INT32, actual_span))

    # Add encoded data
    args.extend(encoded_data)

    # Build kwargs dict for attributes
    kwargs: Dict[str, Any] = {"target_memory": target_memory}

    return _ir_core.create_op_call("block.loadex", args, kwargs, actual_span)
