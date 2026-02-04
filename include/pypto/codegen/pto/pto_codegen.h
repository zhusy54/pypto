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

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

/**
 * @brief PTO MLIR code generator
 *
 * Generates PTO-ISA MLIR format code from PyPTO IR Program.
 * Traverses the IR using the visitor pattern (aligned with CCECodegen).
 * Automatically generates make_tensor_view, subview, and alloc_tile instructions.
 */
class PTOCodegen : public ir::IRVisitor {
 public:
  PTOCodegen() = default;
  ~PTOCodegen() override = default;

  /**
   * @brief Generate PTO-ISA MLIR format code from IR Program
   *
   * @param program Input PyPTO IR Program
   * @return MLIR code as string
   */
  std::string Generate(const ir::ProgramPtr& program);

  // Public helper methods for operator codegen functions

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
   * @brief Get or emit float constant (emits to constants section, returns SSA name)
   *
   * @param value Constant value
   * @param mlir_type MLIR type string (e.g., "f32", "i32")
   * @return SSA variable name for the constant
   */
  std::string GetOrEmitFloatConstant(double value, const std::string& mlir_type = "f32");

  /**
   * @brief Emit a line of MLIR code
   *
   * @param mlir_code MLIR code line to emit
   */
  void EmitMLIR(const std::string& mlir_code);

 protected:
  // Override visitor methods for code generation - Statements
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::SeqStmtsPtr& op) override;
  void VisitStmt_(const ir::OpStmtsPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;

  // Override visitor methods for code generation - Expressions
  void VisitExpr_(const ir::CallPtr& op) override;

 private:
  /**
   * @brief Generate PTO-ISA MLIR for a single function
   */
  void GenerateFunction(const ir::FunctionPtr& func);

  /**
   * @brief Build variable name to MemRef mapping from function body
   */
  void BuildVarToMemRefMapping(const ir::FunctionPtr& func);

  /**
   * @brief Emit make_tensor_view for all tensor parameters
   */
  void EmitMakeTensorViews(const ir::FunctionPtr& func);

  /**
   * @brief Emit alloc_tile for all MemRefs
   */
  void EmitAllocTiles(const ir::FunctionPtr& func, const std::vector<ir::MemRefPtr>& memrefs);

  /**
   * @brief Emit subview part of block.load (tload is emitted in AssignStmt visitor)
   */
  void EmitBlockLoadSubview(const ir::CallPtr& op);

  /**
   * @brief Emit block.store -> subview + tstore
   */
  void EmitBlockStore(const ir::CallPtr& op);

  /**
   * @brief Get indent string for current level
   */
  std::string GetIndent() const;

  /**
   * @brief Get or emit index constant (internal; writes to constants section)
   */
  std::string GetOrEmitIndexConstant(int64_t value);

  /**
   * @brief Get tile_buf name for a MemRef
   */
  std::string GetTileBufForMemRef(const ir::MemRefPtr& memref);

  // Output streams
  std::ostringstream stream_;
  std::ostringstream constants_section_;
  std::ostringstream body_section_;
  int indent_level_ = 0;

  // Variable mappings
  std::map<std::string, std::string> var_to_mlir_;
  std::map<std::string, std::string> tensor_to_view_;
  std::map<const ir::MemRef*, std::string> memref_to_mlir_;
  std::map<std::string, const ir::MemRef*> var_to_memref_;
  std::set<int64_t> emitted_constants_;
  std::set<double> emitted_float_constants_;
  std::map<double, std::string> float_const_names_;

  int temp_counter_ = 0;

  // Current function context
  ir::FunctionPtr current_function_;
  std::string current_tile_view_;
  std::string current_result_buf_;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
