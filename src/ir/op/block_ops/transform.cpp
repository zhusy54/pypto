/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file transform.cpp
 * @brief Shape transformation block operations (view, reshape, transpose)
 *
 * This file implements shape transformation operations for tiles including
 * view, reshape and transpose operations.
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {
// ============================================================================
// Helper Functions (file-local)
// ============================================================================

/**
 * @brief Normalize axis index to handle negative indexing
 *
 * @param axis The axis index (can be negative)
 * @param ndim The number of dimensions
 * @return The normalized axis index
 */
int NormalizeAxis(int axis, size_t ndim) {
  if (axis < 0) {
    axis += static_cast<int>(ndim);
  }
  CHECK(axis >= 0 && axis < static_cast<int>(ndim))
      << "Axis " << axis << " is out of range for " << ndim << "D tile";
  return axis;
}

/**
 * @brief Compute the product of shape dimensions (for static shapes)
 *
 * @param shape The shape dimensions
 * @return The product if all dimensions are ConstInt, -1 otherwise
 */
int64_t ComputeShapeProduct(const std::vector<ExprPtr>& shape) {
  int64_t product = 1;
  for (const auto& dim : shape) {
    auto const_dim = As<ConstInt>(dim);
    if (!const_dim) {
      return -1;  // Dynamic shape, cannot compute product
    }
    product *= const_dim->value_;
  }
  return product;
}

/**
 * @brief Layout operation types for block.loadex
 */
enum class LayoutOpTypeEnum : int {
  VIEW = 0,
  RESHAPE = 1,
  TRANSPOSE = 2,
};

/**
 * @brief Decoded layout operation with type and parameters
 */
struct DecodedOp {
  LayoutOpTypeEnum type;
  std::vector<ExprPtr> params;
};

/**
 * @brief Decode flattened layout operations from args
 *
 * @param args The flattened arguments from block.loadex call
 * @return Vector of decoded operations
 *
 * Encoding format:
 *   args[0]: tile (input)
 *   args[1]: num_ops (ConstInt)
 *   args[2]: encoding_length (ConstInt)
 *   args[3..]: [op_type, param_count, param1, param2, ...]
 */
std::vector<DecodedOp> DecodeLayoutOps(const std::vector<ExprPtr>& args) {
  CHECK(args.size() >= 3) << "block.loadex requires at least 3 arguments (tile, num_ops, encoding_length)";

  // Extract num_ops
  auto num_ops_const = As<ConstInt>(args[1]);
  CHECK(num_ops_const) << "block.loadex: num_ops must be ConstInt";
  int num_ops = static_cast<int>(num_ops_const->value_);
  CHECK(num_ops > 0) << "block.loadex: num_ops must be positive, got " << num_ops;

  // Extract encoding_length
  auto encoding_length_const = As<ConstInt>(args[2]);
  CHECK(encoding_length_const) << "block.loadex: encoding_length must be ConstInt";
  int encoding_length = static_cast<int>(encoding_length_const->value_);
  CHECK(encoding_length > 0) << "block.loadex: encoding_length must be positive, got " << encoding_length;

  // Verify total argument count
  CHECK(args.size() == static_cast<size_t>(3 + encoding_length))
      << "block.loadex: expected " << (3 + encoding_length) << " arguments, but got " << args.size();

  std::vector<DecodedOp> ops;
  ops.reserve(num_ops);

  size_t pos = 3;  // Start after tile, num_ops, encoding_length

  for (int i = 0; i < num_ops; ++i) {
    CHECK(pos < args.size()) << "block.loadex: insufficient encoded data for operation " << i;

    // Read op_type
    auto op_type_const = As<ConstInt>(args[pos++]);
    CHECK(op_type_const) << "block.loadex: op_type must be ConstInt at position " << (pos - 1);
    int op_type_val = static_cast<int>(op_type_const->value_);
    CHECK(op_type_val >= 0 && op_type_val <= 2)
        << "block.loadex: invalid op_type " << op_type_val << " (must be 0=VIEW, 1=RESHAPE, 2=TRANSPOSE)";
    LayoutOpTypeEnum op_type = static_cast<LayoutOpTypeEnum>(op_type_val);

    // Read param_count
    CHECK(pos < args.size()) << "block.loadex: missing param_count for operation " << i;
    auto param_count_const = As<ConstInt>(args[pos++]);
    CHECK(param_count_const) << "block.loadex: param_count must be ConstInt at position " << (pos - 1);
    int param_count = static_cast<int>(param_count_const->value_);
    CHECK(param_count > 0) << "block.loadex: param_count must be positive, got " << param_count;

    // Read parameters
    std::vector<ExprPtr> params;
    params.reserve(param_count);
    for (int j = 0; j < param_count; ++j) {
      CHECK(pos < args.size()) << "block.loadex: insufficient parameters for operation " << i << " (expected "
                               << param_count << ", got " << j << ")";
      params.push_back(args[pos++]);
    }

    ops.push_back(DecodedOp{op_type, std::move(params)});
  }

  return ops;
}

}  // anonymous namespace

// ============================================================================
// Type Inference Functions
// ============================================================================

TypePtr DeduceTileViewType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.view requires at least 2 arguments: input tile and shape_ndim
  // Followed by shape dimensions and offset dimensions
  CHECK(args.size() >= 2) << "tile.view requires at least 2 arguments (input, shape_ndim), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.view requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument is the number of shape dimensions (ConstInt)
  auto shape_ndim_const = As<ConstInt>(args[1]);
  CHECK(shape_ndim_const)
      << "tile.view requires second argument to be a ConstInt indicating number of shape dimensions";

  size_t shape_ndim = static_cast<size_t>(shape_ndim_const->value_);
  CHECK(shape_ndim > 0) << "tile.view requires at least 1 shape dimension";
  CHECK(shape_ndim <= 2) << "tile.view: TileType supports at most 2 dimensions, but got " << shape_ndim;

  // Check we have enough arguments: input + shape_ndim + shape_dims + offset_dims
  CHECK(args.size() >= 2 + shape_ndim)
      << "tile.view requires at least " << (2 + shape_ndim) << " arguments for shape_ndim=" << shape_ndim
      << ", but got " << args.size();

  // Extract new shape dimensions (args[2] to args[2 + shape_ndim - 1])
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_ndim);
  for (size_t i = 0; i < shape_ndim; ++i) {
    new_shape.emplace_back(args[2 + i]);
  }

  // The remaining arguments are offset dimensions (not used for type deduction)
  // View preserves dtype but has new shape (which can have different rank than input)
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

TypePtr DeduceTileReshapeType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.reshape requires at least 2 arguments: input tile and shape_ndim
  // Followed by shape dimensions
  CHECK(args.size() >= 2) << "tile.reshape requires at least 2 arguments (input, shape_ndim), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.reshape requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument is the number of shape dimensions (ConstInt)
  auto shape_ndim_const = As<ConstInt>(args[1]);
  CHECK(shape_ndim_const)
      << "tile.reshape requires second argument to be a ConstInt indicating number of shape dimensions";

  size_t shape_ndim = static_cast<size_t>(shape_ndim_const->value_);
  CHECK(shape_ndim > 0) << "tile.reshape requires at least 1 shape dimension";
  CHECK(shape_ndim <= 2) << "tile.reshape: TileType supports at most 2 dimensions, but got " << shape_ndim;

  // Check we have correct number of arguments: input + shape_ndim + shape_dims
  CHECK(args.size() == 2 + shape_ndim)
      << "tile.reshape requires exactly " << (2 + shape_ndim) << " arguments for shape_ndim=" << shape_ndim
      << ", but got " << args.size();

  // Extract new shape dimensions (args[2] to args[2 + shape_ndim - 1])
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_ndim);
  for (size_t i = 0; i < shape_ndim; ++i) {
    new_shape.emplace_back(args[2 + i]);
  }

  // For static shapes, verify that the total number of elements matches
  int64_t old_product = ComputeShapeProduct(tile_type->shape_);
  int64_t new_product = ComputeShapeProduct(new_shape);

  if (old_product > 0 && new_product > 0) {
    CHECK(old_product == new_product) << "tile.reshape: cannot reshape tile of size " << old_product
                                      << " into shape with size " << new_product;
  }

  // Return new TileType with reshaped dimensions and same dtype
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

TypePtr DeduceTileTransposeType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tile.transpose requires exactly 3 arguments: input tile, axis1, axis2
  CHECK(args.size() == 3) << "tile.transpose requires exactly 3 arguments (input, axis1, axis2), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.transpose requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  const auto& input_shape = tile_type->shape_;
  size_t ndim = input_shape.size();

  CHECK(ndim == 2) << "tile.transpose requires exactly 2 dimensions (TileType constraint), but got " << ndim;

  // Second argument is axis1 (ConstInt)
  auto axis1_const = As<ConstInt>(args[1]);
  CHECK(axis1_const) << "tile.transpose requires second argument (axis1) to be a ConstInt";

  // Third argument is axis2 (ConstInt)
  auto axis2_const = As<ConstInt>(args[2]);
  CHECK(axis2_const) << "tile.transpose requires third argument (axis2) to be a ConstInt";

  // Normalize axes (handle negative indexing)
  int axis1 = NormalizeAxis(static_cast<int>(axis1_const->value_), ndim);
  int axis2 = NormalizeAxis(static_cast<int>(axis2_const->value_), ndim);

  CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=" << axis1
                        << ", axis2=" << axis2;

  // Create new shape by swapping the specified dimensions
  std::vector<ExprPtr> new_shape = input_shape;
  std::swap(new_shape[axis1], new_shape[axis2]);

  // Return new TileType with transposed shape and same dtype
  return std::make_shared<TileType>(new_shape, tile_type->dtype_);
}

TypePtr DeduceTileLoadexType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // block.loadex signature: (tensor, num_ops, encoding_length, encoded_data...)
  CHECK(args.size() >= 3) << "block.loadex requires at least 3 arguments (tensor, num_ops, encoding_length)";

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "block.loadex requires first argument to be TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Extract num_ops from args[1]
  auto num_ops_const = As<ConstInt>(args[1]);
  CHECK(num_ops_const) << "block.loadex: num_ops must be ConstInt";
  int num_ops = static_cast<int>(num_ops_const->value_);
  CHECK(num_ops > 0) << "block.loadex: num_ops must be positive, got " << num_ops;

  // Extract encoding_length from args[2]
  auto encoding_length_const = As<ConstInt>(args[2]);
  CHECK(encoding_length_const) << "block.loadex: encoding_length must be ConstInt";
  int encoding_length = static_cast<int>(encoding_length_const->value_);
  CHECK(encoding_length > 0) << "block.loadex: encoding_length must be positive, got " << encoding_length;

  // Verify total argument count
  CHECK(args.size() == static_cast<size_t>(3 + encoding_length))
      << "block.loadex: expected " << (3 + encoding_length) << " arguments, but got " << args.size();

  // Decode operations from flattened args (starting at position 3)
  std::vector<ExprPtr> encoded_args(args.begin() + 3, args.end());

  // Temporarily insert a placeholder for decoding - we need num_ops, encoding_length, and encoded data
  std::vector<ExprPtr> decode_args;
  decode_args.push_back(nullptr);  // Placeholder for tile (not used in DecodeLayoutOps)
  decode_args.push_back(args[1]);  // num_ops
  decode_args.push_back(args[2]);  // encoding_length
  decode_args.insert(decode_args.end(), encoded_args.begin(), encoded_args.end());

  auto ops = DecodeLayoutOps(decode_args);

  // Determine initial tile shape:
  // - If first op is VIEW: use VIEW's shape as the load region
  // - Otherwise: load entire tensor
  TypePtr current_type;
  size_t ops_start_idx = 0;

  if (!ops.empty() && ops[0].type == LayoutOpTypeEnum::VIEW) {
    // First operation is VIEW - use its shape for loading
    const auto& view_params = ops[0].params;
    CHECK(view_params.size() % 2 == 0)
        << "block.loadex: VIEW operation must have even number of params (shape + offset)";

    int shape_ndim = static_cast<int>(view_params.size() / 2);
    std::vector<ExprPtr> load_shape(view_params.begin(), view_params.begin() + shape_ndim);

    current_type = std::make_shared<TileType>(load_shape, tensor_type->dtype_);
    ops_start_idx = 1;  // Skip the first VIEW operation (it's used for loading)
  } else {
    // No VIEW as first operation - load entire tensor
    current_type = std::make_shared<TileType>(tensor_type->shape_, tensor_type->dtype_);
    ops_start_idx = 0;  // Process all operations
  }

  // Apply remaining type transformations sequentially
  for (size_t i = ops_start_idx; i < ops.size(); ++i) {
    const auto& op = ops[i];

    // Create a temporary Var with current_type as a placeholder for type deduction
    auto temp_var = std::make_shared<Var>("_loadex_temp", current_type, Span::unknown());

    std::vector<ExprPtr> reconstructed_args;
    reconstructed_args.push_back(temp_var);  // Use temp var instead of nullptr

    switch (op.type) {
      case LayoutOpTypeEnum::VIEW: {
        // VIEW params: [shape..., offset...]
        // We need to reconstruct args: [tile, shape_ndim, shape..., offset...]
        CHECK(op.params.size() % 2 == 0)
            << "block.loadex: VIEW operation " << i
            << " must have even number of params (shape + offset), got " << op.params.size();

        int shape_ndim = static_cast<int>(op.params.size() / 2);
        reconstructed_args.push_back(
            std::make_shared<ConstInt>(shape_ndim, DataType::INT32, Span::unknown()));

        // Add all params (shape followed by offset)
        reconstructed_args.insert(reconstructed_args.end(), op.params.begin(), op.params.end());

        current_type = DeduceTileViewType(reconstructed_args, kwargs);
        break;
      }

      case LayoutOpTypeEnum::RESHAPE: {
        // RESHAPE params: [shape...]
        // Reconstruct args: [tile, shape_ndim, shape...]
        int shape_ndim = static_cast<int>(op.params.size());
        reconstructed_args.push_back(
            std::make_shared<ConstInt>(shape_ndim, DataType::INT32, Span::unknown()));

        // Add shape params
        reconstructed_args.insert(reconstructed_args.end(), op.params.begin(), op.params.end());

        current_type = DeduceTileReshapeType(reconstructed_args, kwargs);
        break;
      }

      case LayoutOpTypeEnum::TRANSPOSE: {
        // TRANSPOSE params: [axis1, axis2]
        // Reconstruct args: [tile, axis1, axis2]
        CHECK(op.params.size() == 2) << "block.loadex: TRANSPOSE operation " << i
                                     << " requires exactly 2 params (axis1, axis2), got " << op.params.size();

        reconstructed_args.push_back(op.params[0]);
        reconstructed_args.push_back(op.params[1]);

        current_type = DeduceTileTransposeType(reconstructed_args, kwargs);
        break;
      }

      default:
        throw ValueError("block.loadex: unknown operation type");
    }
  }

  return current_type;
}

// ============================================================================
// Registration Function for Tile Transform Operations
// ============================================================================

REGISTER_OP("block.view")
    .set_op_category("BlockOp")
    .set_description("Create a view/slice of a tile with new shape and offset")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape_ndim", "Number of shape dimensions (ConstInt)")
    .add_argument("shape_dims", "New shape dimensions (variable number)")
    .add_argument("offset_dims", "Offset dimensions (variable number)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileViewType(args, kwargs);
    });

REGISTER_OP("block.reshape")
    .set_op_category("BlockOp")
    .set_description("Reshape tile to new shape")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("shape_ndim", "Number of shape dimensions (ConstInt)")
    .add_argument("shape_dims", "New shape dimensions (variable number)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReshapeType(args, kwargs);
    });

REGISTER_OP("block.transpose")
    .set_op_category("BlockOp")
    .set_description("Transpose tile by swapping two axes")
    .add_argument("input", "Input tile (TileType)")
    .add_argument("axis1", "First axis to swap (ConstInt)")
    .add_argument("axis2", "Second axis to swap (ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTransposeType(args, kwargs);
    });

REGISTER_OP("block.loadex")
    .set_op_category("BlockOp")
    .set_description("Load data from tensor with layout transformations applied")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("num_ops", "Number of operations (ConstInt)")
    .add_argument("encoding_length", "Length of encoded data (ConstInt)")
    .add_argument("encoded_data", "Flattened operation data (variable length)")
    .set_attr<int>("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileLoadexType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
