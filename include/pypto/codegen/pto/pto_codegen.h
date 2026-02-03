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

#ifndef PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
#define PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_

#include <memory>
#include <string>

#include "pypto/core/dtype.h"

namespace pypto {
namespace ir {
// Forward declarations
class Program;
using ProgramPtr = std::shared_ptr<const Program>;
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;
class Var;
using VarPtr = std::shared_ptr<const Var>;
}  // namespace ir

namespace codegen {

// Forward declare internal implementation class
class PTOMLIRCodegen;

/**
 * @brief PTO MLIR code generator
 *
 * Generates PTO-ISA MLIR format code from PyPTO IR Program.
 * Automatically generates make_tensor_view, subview, and alloc_tile instructions.
 */
class PTOCodegen {
 public:
  PTOCodegen() = default;
  ~PTOCodegen() = default;

  /**
   * @brief Generate PTO-ISA MLIR format code from IR Program
   *
   * @param program Input PyPTO IR Program
   * @return MLIR code as string
   */
  std::string Generate(const ir::ProgramPtr& program);

  // Public helper methods for operator codegen functions
  // These forward to the internal implementation

  /**
   * @brief Create a new temporary SSA variable
   *
   * @return New SSA variable name (e.g., "%1", "%2")
   */
  std::string NewTemp();

  /**
   * @brief Get the current result buffer for tile operations
   *
   * @return Current result buffer name
   */
  [[nodiscard]] std::string GetCurrentResultBuf() const;

  /**
   * @brief Get MLIR SSA variable for an expression
   *
   * @param expr Expression to get SSA var for
   * @return MLIR SSA variable name
   */
  std::string GetMLIRVar(const ir::ExprPtr& expr);

  /**
   * @brief Extract constant integer value from expression
   *
   * @param expr Expression (must be ConstInt)
   * @return Integer value
   */
  int64_t GetConstIntValue(const ir::ExprPtr& expr);

  /**
   * @brief Get or create tensor view for a variable
   *
   * @param tensor Tensor variable
   * @return Tensor view name
   */
  std::string GetOrCreateTensorView(const ir::VarPtr& tensor);

  /**
   * @brief Convert DataType to MLIR type string
   *
   * @param dtype Data type
   * @return MLIR type string (e.g., "f32", "i32")
   */
  std::string DataTypeToMLIR(const DataType& dtype);

  /**
   * @brief Get or emit index constant
   *
   * @param val Constant value
   * @return Index constant string
   */
  std::string GetIndexConstant(int64_t val);

  /**
   * @brief Emit a line of MLIR code
   *
   * @param mlir_code MLIR code line to emit
   */
  void EmitMLIR(const std::string& mlir_code);

 private:
  // Pointer to current implementation (set during Generate())
  // Uses void* to avoid exposing internal PTOMLIRCodegen class
  void* current_impl_ = nullptr;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
