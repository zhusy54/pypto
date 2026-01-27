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

#include "pypto/ir/transform/add_alloc_pass.h"

#include <memory>
#include <set>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }

  void VisitExpr_(const VarPtr& op) override {
    // Check if this variable has a TileType with MemRef
    auto tile_type = std::dynamic_pointer_cast<const TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value());
    }
  }

  void VisitExpr_(const IterArgPtr& op) override {
    // Check if this iteration argument has a TileType with MemRef
    auto tile_type = std::dynamic_pointer_cast<const TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value());
    }
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const MemRef*> seen_ptrs_;  // Track raw MemRef pointers to avoid duplicates

  void AddMemRefIfUnique(const MemRefPtr& memref) {
    // Use raw pointer address to check uniqueness (same shared_ptr)
    const MemRef* raw_ptr = memref.get();
    if (seen_ptrs_.find(raw_ptr) == seen_ptrs_.end()) {
      memrefs_.push_back(memref);
      seen_ptrs_.insert(raw_ptr);
    }
  }
};

}  // namespace

void AddAllocPass::CollectMemRefsFromStatement(const StmtPtr& stmt, std::vector<MemRefPtr>& memrefs) {
  // Create a visitor to traverse the statement
  MemRefCollectorVisitor visitor;
  visitor.VisitStmt(stmt);

  // Add collected MemRefs to the vector (avoiding duplicates by comparing raw pointers)
  std::set<const ir::MemRef*> existing_ptrs;
  for (const auto& mr : memrefs) {
    existing_ptrs.insert(mr.get());
  }

  for (const auto& mr : visitor.GetMemRefs()) {
    if (existing_ptrs.find(mr.get()) == existing_ptrs.end()) {
      memrefs.push_back(mr);
      existing_ptrs.insert(mr.get());
    }
  }
}

FunctionPtr AddAllocPass::Run(const FunctionPtr& func) {
  // Step 1: Collect all unique MemRef objects from TileType variables in the function
  std::vector<MemRefPtr> memrefs;
  CollectMemRefsFromStatement(func->body_, memrefs);

  // Step 2: Create alloc operations for each MemRef
  std::vector<StmtPtr> alloc_stmts;

  for (const auto& memref : memrefs) {
    // Create block.alloc operation with memref ID as argument
    auto alloc_op = std::make_shared<Op>("block.alloc");

    // Create a ConstInt expression for the MemRef ID
    auto memref_id_expr = std::make_shared<ConstInt>(
        static_cast<int64_t>(memref->id_),
        DataType::INT64,
        Span::unknown()
    );

    std::vector<ExprPtr> alloc_args;
    alloc_args.push_back(memref_id_expr);

    // Create a Call expression for the alloc operation
    // The alloc operation returns INT64 (pointer address)
    auto alloc_call = std::make_shared<Call>(
        alloc_op,
        alloc_args,
        std::make_shared<ScalarType>(DataType::INT64),
        Span::unknown()
    );

    // Create a temporary variable to hold the result (allocated pointer)
    auto memref_var_name = "alloc_" + std::to_string(memref->id_);
    auto memref_var = std::make_shared<Var>(
        memref_var_name,
        std::make_shared<ScalarType>(DataType::INT64),
        Span::unknown()
    );

    // Create an assignment statement: alloc_var = block.alloc(memref_id)
    auto assign_stmt = std::make_shared<AssignStmt>(memref_var, alloc_call, Span::unknown());
    alloc_stmts.push_back(assign_stmt);
  }

  // Step 3: Prepend alloc statements to function body
  StmtPtr new_body = func->body_;

  if (!alloc_stmts.empty()) {
    // If there are alloc statements, create a sequence
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(func->body_)) {
      // Append alloc statements before existing statements
      std::vector<StmtPtr> all_stmts = alloc_stmts;
      all_stmts.insert(all_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
      new_body = std::make_shared<SeqStmts>(all_stmts, func->body_->span_);
    } else {
      // Wrap existing body in sequence with alloc statements
      alloc_stmts.push_back(func->body_);
      new_body = std::make_shared<SeqStmts>(alloc_stmts, func->body_->span_);
    }
  }

  // Step 4: Return transformed function
  return std::make_shared<Function>(func->name_, func->params_, func->return_types_, new_body, func->span_);
}

}  // namespace ir
}  // namespace pypto
