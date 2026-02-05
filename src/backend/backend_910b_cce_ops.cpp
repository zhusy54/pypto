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
 * @file backend_910b_cce_ops.cpp
 * @brief Backend op registration for Backend910B_CCE
 *
 * This file registers all block operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <string>

#include "pypto/backend/backend_910b_cce.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/pipe.h"

namespace pypto {
namespace backend {

// ============================================================================
// Matmul Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 2) << "block.matmul requires 2 arguments: lhs, rhs";

      std::string lhs = codegen.GetExprAsCode(op->args_[0]);
      std::string rhs = codegen.GetExprAsCode(op->args_[1]);
      std::string dst = codegen.GetCurrentResultTarget();

      codegen.Emit("TMATMUL(" + dst + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 3) << "block.matmul_acc requires 3 arguments: acc, lhs, rhs";

      std::string acc = codegen.GetExprAsCode(op->args_[0]);
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst = codegen.GetCurrentResultTarget();

      // TMATMUL_ACC accumulates into dst, which should be initialized from acc
      codegen.Emit("TMATMUL_ACC(" + dst + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

}  // namespace backend
}  // namespace pypto
