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

#include "pypto/passes/data_dependency_analysis_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/transform/base/visitor.h"

namespace pypto {
namespace ir {

FunctionPtr DataDependencyAnalysisPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "DataDependencyAnalysisPass cannot run on null function";

  // Step 1: Identify basic blocks from control flow
  std::vector<BasicBlock> blocks = IdentifyBasicBlocks(func->body_);
  LOG_INFO << "Identified " << blocks.size() << " basic blocks";

  // Step 2: Build dependency graph for each basic block
  std::vector<DependencyEdge> all_dependencies;
  for (const auto& block : blocks) {
    auto block_deps = AnalyzeBlockDependencies(block);
    all_dependencies.insert(all_dependencies.end(), block_deps.begin(), block_deps.end());
  }
  LOG_INFO << "Found " << all_dependencies.size() << " dependency edges";

  // Step 3: Annotate function with dependency graph and basic block metadata
  FunctionPtr annotated_func = AnnotateWithDependencies(func, blocks, all_dependencies);

  return annotated_func;
}

std::vector<BasicBlock> DataDependencyAnalysisPass::IdentifyBasicBlocks(const StmtPtr& stmt) {
  std::vector<BasicBlock> blocks;

  // TODO: Implement basic block identification
  // Strategy:
  // 1. Traverse control flow graph
  // 2. Identify block boundaries:
  //    - Start of function
  //    - After branch/loop (end of IfStmt/ForStmt)
  //    - Branch/loop targets (start of then_body, else_body, loop body)
  // 3. For each block:
  //    - Assign unique ID
  //    - Collect statements in the block
  //    - Identify predecessors and successors
  //    - Mark if it's a loop body (is_loop_body = true for ForStmt body)
  //
  // Example structure:
  // BasicBlock bb;
  // bb.id = next_id++;
  // bb.statements = [...];
  // bb.predecessors = [...];
  // bb.successors = [...];
  // bb.is_loop_body = false;
  // blocks.push_back(bb);

  return blocks;
}

std::vector<DependencyEdge> DataDependencyAnalysisPass::AnalyzeBlockDependencies(const BasicBlock& block) {
  std::vector<DependencyEdge> dependencies;

  // TODO: Implement dependency analysis for a basic block
  // Strategy:
  // 1. Build dataflow graph of operations in the block
  // 2. Track last write and last read for each variable
  // 3. For each operation:
  //    - Identify variables read (inputs)
  //    - Identify variables written (outputs)
  //    - For each read variable:
  //      - If there's a last write: add RAW dependency
  //    - For each written variable:
  //      - If there's a last read: add WAR dependency
  //      - If there's a last write: add WAW dependency
  //    - Extract pipe types using GetPipeType()
  //
  // Example edge creation:
  // DependencyEdge edge;
  // edge.producer = producer_stmt;
  // edge.consumer = consumer_stmt;
  // edge.variable = var;
  // edge.type = DependencyEdge::RAW;
  // edge.producer_pipe = GetPipeType(producer_call);
  // edge.consumer_pipe = GetPipeType(consumer_call);
  // dependencies.push_back(edge);

  return dependencies;
}

std::vector<DependencyEdge> DataDependencyAnalysisPass::MergeDependencies(
    const std::vector<std::vector<DependencyEdge>>& path_dependencies) {
  std::vector<DependencyEdge> merged;

  // TODO: Implement conservative dependency merging for control flow
  // Strategy:
  // 1. Take union of all dependencies from all paths
  // 2. Remove duplicates (same producer, consumer, variable, type)
  // 3. This is conservative: assumes all paths may be taken
  //
  // For IfStmt: merge dependencies from then_body and else_body
  // For ForStmt: include both iteration-internal and loop-carried dependencies

  return merged;
}

std::string DataDependencyAnalysisPass::GetPipeType(const CallPtr& call_expr) {
  // TODO: Extract pipe type from block operation metadata
  // Strategy:
  // 1. Get the operation name from call_expr
  // 2. Look up the op registration
  // 3. Read pipe_type attribute: op->get_attr<std::string>("pipe_type")
  // 4. Return one of: "CUBE", "VECTOR", "MTE1", "MTE2", "MTE3", "SCALAR"
  //
  // Note: Pipe type is already set during op registration via set_attr

  return "UNKNOWN";  // Placeholder
}

FunctionPtr DataDependencyAnalysisPass::AnnotateWithDependencies(const FunctionPtr& func,
                                                                  const std::vector<BasicBlock>& blocks,
                                                                  const std::vector<DependencyEdge>& dependencies) {
  // TODO: Annotate function with dependency graph metadata
  // Strategy:
  // 1. Store basic blocks and dependencies as function attributes
  // 2. These will be read by subsequent passes (MemoryReuseAnalysisPass, OutOfOrderSchedulingPass)
  // 3. Options:
  //    - Add fields to Function class (requires modifying ir/function.h)
  //    - Use a separate annotation map (std::unordered_map<FunctionPtr, Metadata>)
  //    - Store in function body as special statement types
  //
  // For now, return function as-is (metadata storage mechanism TBD)

  return func;  // Placeholder
}

}  // namespace ir
}  // namespace pypto
