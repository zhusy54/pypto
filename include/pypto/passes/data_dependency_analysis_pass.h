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

#ifndef PYPTO_PASSES_DATA_DEPENDENCY_ANALYSIS_PASS_H_
#define PYPTO_PASSES_DATA_DEPENDENCY_ANALYSIS_PASS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transform/base/pass.h"

namespace pypto {
namespace ir {

/**
 * @brief Represents a basic block in the control flow graph
 */
struct BasicBlock {
  int id;                                // Unique identifier
  std::vector<StmtPtr> statements;       // Statements in this block
  std::vector<int> predecessors;         // Predecessor block IDs
  std::vector<int> successors;           // Successor block IDs
  bool is_loop_body = false;             // True if this block is a loop body
};

/**
 * @brief Represents a data dependency edge in the dependency graph
 */
struct DependencyEdge {
  enum Type { RAW, WAR, WAW };           // Read-After-Write, Write-After-Read, Write-After-Write

  StmtPtr producer;                      // Producer statement
  StmtPtr consumer;                      // Consumer statement
  VarPtr variable;                       // Variable creating the dependency
  Type type;                             // Dependency type
  std::string producer_pipe;             // Pipe type of producer (e.g., "VECTOR")
  std::string consumer_pipe;             // Pipe type of consumer
};

/**
 * @brief Pass for analyzing data dependencies between block operations
 *
 * This pass builds a dependency graph and identifies basic blocks. It handles
 * control flow conservatively: merging dependencies from all possible paths.
 *
 * Control flow handling (V1 - Basic Block Level):
 * - If-Else: Merge dependencies from both branches (union)
 * - For loops: Identify loop-carried dependencies, add virtual edges
 * - Basic blocks: Divide CFG into basic blocks for local scheduling
 *
 * Requires: AllocOpInsertionPass to have been run first
 *
 * See plan document "Control Flow Handling Strategy" section for details.
 */
class DataDependencyAnalysisPass : public Pass {
 public:
  DataDependencyAnalysisPass() = default;
  ~DataDependencyAnalysisPass() override = default;

  /**
   * @brief Execute the data dependency analysis pass on a function
   *
   * @param func Input function with AllocOp statements
   * @return Function annotated with dependency graph and basic block metadata
   */
  FunctionPtr Run(const FunctionPtr& func) override;

 private:
  /**
   * @brief Identify basic blocks from control flow
   *
   * @param stmt Function body
   * @return Vector of basic blocks
   */
  std::vector<BasicBlock> IdentifyBasicBlocks(const StmtPtr& stmt);

  /**
   * @brief Build dependency graph for a basic block
   *
   * @param block Basic block to analyze
   * @return Vector of dependency edges
   */
  std::vector<DependencyEdge> AnalyzeBlockDependencies(const BasicBlock& block);

  /**
   * @brief Merge dependencies from multiple control flow paths (conservative)
   *
   * @param path_dependencies Dependencies from each path
   * @return Merged dependencies (union)
   */
  std::vector<DependencyEdge> MergeDependencies(
      const std::vector<std::vector<DependencyEdge>>& path_dependencies);

  /**
   * @brief Extract pipe type from block operation
   *
   * Reads pipe_type from op.get_attr("pipe_type")
   *
   * @param call_expr Call expression (block operation)
   * @return Pipe type string (CUBE, VECTOR, MTE1, MTE2, MTE3, SCALAR)
   */
  std::string GetPipeType(const CallPtr& call_expr);

  /**
   * @brief Annotate function with dependency graph metadata
   *
   * Stores dependency graph as function attribute for use by subsequent passes
   *
   * @param func Original function
   * @param blocks Basic blocks
   * @param dependencies Dependency edges
   * @return Function with annotations
   */
  FunctionPtr AnnotateWithDependencies(const FunctionPtr& func, const std::vector<BasicBlock>& blocks,
                                       const std::vector<DependencyEdge>& dependencies);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_PASSES_DATA_DEPENDENCY_ANALYSIS_PASS_H_
