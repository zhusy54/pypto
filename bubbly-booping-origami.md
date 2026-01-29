# GenerateOrchestrationCode() 实现计划

## 概述

为 PyPTO 的 PTOCodegen 类实现 `GenerateOrchestrationCode()` 方法，该方法将编排函数(Orchestration Function)转换为 C++ Runtime API 代码，用于构建任务图。

## 背景分析

### 输入示例 (dynamic_softmax_codegen.py)

编排函数特征:
- **参数**: TensorType 类型 (代表 DDR 内存中的大张量)
  - 输入张量: `input` [128, 128]
  - 输出张量: `output` [128, 128]
  - 临时张量: `temp_rowmax`, `temp_shifted`, `temp_exp`, `temp_rowsum`
- **函数体**: 包含 Call 语句调用 InCore 函数
  - `ir.Call(ir.GlobalVar("rowmax"), [input_tensor, temp_rowmax])`
  - `ir.Call(ir.GlobalVar("rowexpandsub"), [...])`
- **控制流**: ForStmt 和 IfStmt
  - FOR 循环处理完整 tiles
  - IF 语句处理尾部

### 输出示例 (example_orch.cpp)

生成的 C++ 代码结构:
```cpp
int BuildExampleGraph(Runtime* runtime, uint64_t* args, int arg_count) {
    // 1. 提取主机指针和大小
    void* host_a = reinterpret_cast<void*>(args[0]);
    size_t size_a = static_cast<size_t>(args[3]);

    // 2. 分配设备内存
    void* dev_a = runtime->host_api.DeviceMalloc(size_a);
    runtime->host_api.CopyToDevice(dev_a, host_a, size_a);

    // 3. 记录输出张量
    runtime->RecordTensorPair(host_f, dev_f, size_f);

    // 4. 构建任务图
    uint64_t args_t0[4] = {dev_a, dev_b, dev_c, SIZE};
    int t0 = runtime->add_task(args_t0, 4, func_id, core_type);

    // 5. 添加依赖关系
    runtime->add_successor(t0, t1);

    return 0;
}
```

### 现有实现状态

- 文件: `src/codegen/pto_codegen.cpp:826-835`
- 状态: 仅返回 TODO 占位符
- 相关方法:
  - `IsOrchestrationFunction()` - 已实现,识别编排函数
  - `Generate()` - 已调用 `GenerateOrchestrationCode()`

## 实现方案

### 整体架构

```
GenerateOrchestrationCode(FunctionPtr func)
  ├─ 1. 生成函数签名和参数验证
  ├─ 2. 提取参数声明 (host pointers + sizes)
  ├─ 3. 生成设备内存分配代码
  ├─ 4. 遍历函数体构建任务图
  │   ├─ 处理 Call 语句 → add_task()
  │   ├─ 处理 ForStmt → 循环内任务
  │   ├─ 处理 IfStmt → 条件任务
  │   └─ 跟踪任务依赖关系 → add_successor()
  └─ 5. 生成返回语句
```

### 关键数据结构

```cpp
// 任务信息
struct TaskInfo {
    std::string task_var;        // 任务变量名 (t0, t1, ...)
    std::string func_name;       // 调用的函数名
    std::vector<std::string> args; // 任务参数
    int func_id;                 // 函数 ID
    int core_type;               // 核类型 (AIC=0, AIV=1)
};

// 张量映射
std::map<std::string, std::string> tensor_to_dev_ptr_;  // tensor_name -> dev_ptr_name
std::map<std::string, std::string> tensor_to_size_;     // tensor_name -> size_name

// 任务依赖跟踪
std::vector<TaskInfo> tasks_;
std::map<std::string, std::vector<std::string>> dependencies_;  // task -> successors
```

### 实现步骤

#### 步骤 1: 函数签名生成

```cpp
std::string func_name = func->name_;
std::string build_func_name = "Build" + CapitalizeFirst(func_name);

oss << "extern \"C\" {\n\n";
oss << "int " << build_func_name << "(Runtime* runtime, uint64_t* args, int arg_count) {\n";
oss << "    // Validate argument count\n";
oss << "    if (arg_count < " << expected_args << ") {\n";
oss << "        std::cerr << \"Error: Expected at least " << expected_args << " args\" << std::endl;\n";
oss << "        return -1;\n";
oss << "    }\n\n";
```

#### 步骤 2: 参数提取

遍历函数参数,为每个 TensorType 生成:
```cpp
// 输入/输出张量: host_ptr + size
void* host_<name> = reinterpret_cast<void*>(args[idx]);
size_t size_<name> = static_cast<size_t>(args[idx+1]);

// 计算 expected_args = params.size() * 2 (每个张量需要 ptr + size)
```

#### 步骤 3: 设备内存分配与张量分类

**张量分类逻辑 (根据用户反馈)**:
- **输入张量**: 作为 Call 参数出现,不是返回值 → 需要 CopyToDevice
- **输出张量**: 作为 Call 返回值出现 → 需要 RecordTensorPair
- **临时张量**: 函数参数但仅用于中间结果 → 只分配内存

**实现方法**:
```cpp
// 1. 收集所有作为 Call 返回值的张量
std::set<std::string> output_tensors;
for (auto call : CollectCallOps(func->body_)) {
    if (auto assign = GetParentAssign(call)) {
        output_tensors.insert(assign->var_->name_);
    }
}

// 2. 为每个参数生成内存操作
for (auto param : func->params_) {
    oss << "    void* dev_" << param->name_
        << " = runtime->host_api.DeviceMalloc(size_" << param->name_ << ");\n";

    if (output_tensors.count(param->name_)) {
        // 输出张量: 记录用于回传
        oss << "    runtime->RecordTensorPair(host_" << param->name_
            << ", dev_" << param->name_ << ", size_" << param->name_ << ");\n";
    } else {
        // 输入张量: 拷贝数据到设备
        oss << "    runtime->host_api.CopyToDevice(dev_" << param->name_
            << ", host_" << param->name_ << ", size_" << param->name_ << ");\n";
    }
}
```

#### 步骤 4: 任务图构建与 CoreType 推断

**CoreType 推断策略 (根据用户要求)**:

用户要求: "根据Function内的任务种类进行判断。当前的task function，只允许存在单一的CoreType，注意要做这个检查"

**实现逻辑**:
1. 当遇到 Call 操作时,查找被调用的 InCore 函数
2. 分析该函数内所有 block 操作,推断核类型
3. 验证所有操作使用相同核类型
4. 使用推断的核类型生成 add_task()

**核类型映射表**:
```cpp
std::map<std::string, CoreType> op_to_core_type_ = {
    // Matrix operations → CUBE (AIC)
    {"block.matmul", CoreType::CUBE},
    {"block.matmul_acc", CoreType::CUBE},

    // Vector operations → VECTOR (AIV)
    {"block.add", CoreType::VECTOR},
    {"block.mul", CoreType::VECTOR},
    {"block.div", CoreType::VECTOR},
    {"block.sub", CoreType::VECTOR},
    {"block.adds", CoreType::VECTOR},
    {"block.muls", CoreType::VECTOR},
    {"block.divs", CoreType::VECTOR},
    {"block.subs", CoreType::VECTOR},
    {"block.exp", CoreType::VECTOR},
    {"block.sqrt", CoreType::VECTOR},
    {"block.row_max", CoreType::VECTOR},
    {"block.row_sum", CoreType::VECTOR},
    {"block.row_expand_add", CoreType::VECTOR},
    {"block.row_expand_sub", CoreType::VECTOR},
    {"block.row_expand_mul", CoreType::VECTOR},
    {"block.row_expand_div", CoreType::VECTOR},

    // Memory operations - 不影响核类型判断
    {"block.load", std::nullopt},
    {"block.store", std::nullopt},
    {"block.alloc", std::nullopt},
};
```

**InferFunctionCoreType 实现**:
```cpp
// 分析 InCore 函数,推断核类型
CoreType InferFunctionCoreType(const FunctionPtr& func, const ProgramPtr& program) {
    std::set<CoreType> core_types;

    // 使用 IRVisitor 收集所有 Call 操作
    class CoreTypeCollector : public IRVisitor {
    public:
        std::set<CoreType> core_types_;
        std::map<std::string, CoreType>& op_map_;

        CoreTypeCollector(std::map<std::string, CoreType>& op_map)
            : op_map_(op_map) {}

        void VisitExpr_(const CallPtr& call) override {
            std::string op_name = call->op_->name_;
            if (op_map_.count(op_name)) {
                auto ct = op_map_[op_name];
                if (ct.has_value()) {  // 忽略 load/store/alloc
                    core_types_.insert(*ct);
                }
            }
            IRVisitor::VisitExpr_(call);
        }
    };

    CoreTypeCollector collector(op_to_core_type_);
    collector.VisitStmt(func->body_);

    // 验证一致性
    CHECK(collector.core_types_.size() <= 1)
        << "Function " << func->name_ << " contains mixed core types. "
        << "All block operations must use the same core type (VECTOR or CUBE).";

    // 默认返回 VECTOR
    if (collector.core_types_.empty()) {
        return CoreType::VECTOR;
    }
    return *collector.core_types_.begin();
}
```

**GenerateTaskFromCall 更新版**:
```cpp
void GenerateTaskFromCall(const CallPtr& call, const ProgramPtr& program) {
    // 1. 提取被调用的函数名
    std::string func_name = call->op_->name_;

    // 2. 查找函数并推断核类型
    auto callee_func = FindFunctionByName(program, func_name);
    CHECK(callee_func) << "Function " << func_name << " not found in program";

    CoreType core_type = InferFunctionCoreType(callee_func, program);
    int func_id = GetOrCreateFuncId(func_name);

    // 3. 生成任务参数数组
    std::vector<std::string> task_args;
    for (const auto& arg : call->args_) {
        if (auto var = As<Var>(arg)) {
            task_args.push_back("dev_" + var->name_);
        } else if (auto const_int = As<ConstInt>(arg)) {
            task_args.push_back(std::to_string(const_int->value_));
        }
    }

    // 4. 生成 add_task 调用
    std::string task_var = "t" + std::to_string(task_counter_++);
    oss << "    // Task " << task_counter_ - 1 << ": Call " << func_name << "\n";
    oss << "    uint64_t args_" << task_var << "[" << task_args.size() << "];\n";
    for (size_t i = 0; i < task_args.size(); ++i) {
        oss << "    args_" << task_var << "[" << i << "] = "
            << "reinterpret_cast<uint64_t>(" << task_args[i] << ");\n";
    }
    oss << "    int " << task_var << " = runtime->add_task(args_"
        << task_var << ", " << task_args.size()
        << ", " << func_id << ", " << static_cast<int>(core_type) << ");\n\n";

    // 5. 记录任务
    tasks_.push_back({task_var, func_name, task_args, func_id, core_type});
}
```

**备选方案: 从 Call kwargs 读取**:

如果 Call 节点的 kwargs 已包含 `func_id` 和 `device_type`,直接使用:
```cpp
// 尝试从 kwargs 读取
std::optional<int> func_id_opt = GetKwarg<int>(call->kwargs_, "func_id");
std::optional<int> device_type_opt = GetKwarg<int>(call->kwargs_, "device_type");

if (func_id_opt && device_type_opt) {
    // 使用已有的元数据
    func_id = *func_id_opt;
    core_type = static_cast<CoreType>(*device_type_opt);
} else {
    // 推断核类型
    auto callee_func = FindFunctionByName(program, func_name);
    core_type = InferFunctionCoreType(callee_func, program);
    func_id = GetOrCreateFuncId(func_name);
}
```

#### 步骤 5: 依赖关系分析

使用数据流分析跟踪依赖:
```cpp
// 简化版本: 按顺序依赖
for (size_t i = 1; i < tasks_.size(); ++i) {
    oss << "    runtime->add_successor(" << tasks_[i-1].task_var
        << ", " << tasks_[i].task_var << ");\n";
}
```

**更精确的版本**: 分析变量使用关系
- 跟踪每个变量的生产者任务 (producer)
- 跟踪每个变量的消费者任务 (consumer)
- 生成 add_successor(producer, consumer)

#### 步骤 6: 函数结尾

```cpp
oss << "    std::cout << \"Created runtime with \" << runtime->get_task_count() "
    << "<< \" tasks\\n\";\n";
oss << "    runtime->print_runtime();\n";
oss << "    return 0;\n";
oss << "}\n\n";
oss << "}  // extern \"C\"\n";
```

### 简化实现 vs 完整实现

#### 简化版本 (MVP - Minimum Viable Product)

**假设**:
- 所有任务按顺序依赖: t0 → t1 → t2 → ...
- 所有函数都是 AIV 核类型
- func_id 使用函数名的哈希值
- 不处理 ForStmt 和 IfStmt (或仅生成注释)
- 所有张量都作为输入+输出处理

**优势**: 快速实现,验证框架正确性

#### 完整版本

**实现**:
- 精确的数据流依赖分析
- 支持 ForStmt 循环生成重复任务
- 支持 IfStmt 条件分支
- 根据函数名推断 core_type
- 智能区分输入/输出张量
- func_id 映射表管理

**优势**: 完整功能,生成最优任务图

### 最终方案: 分阶段实现 (用户选择)

根据用户反馈,采用以下具体实现策略:

**用户决策汇总**:
1. **实现策略**: 分阶段实现 - MVP + 后续迭代
2. **func_id 管理**: 使用全局函数名到 ID 的映射表
3. **CoreType 判断**: 分析函数内所有 block 操作,检查核类型一致性
4. **张量分类**: Call 的参数为输入,返回值为输出

**阶段 1: MVP 实现 (当前任务)**

**核心功能**:
- 生成 C++ 函数签名: `int Build<FuncName>(Runtime* runtime, uint64_t* args, int arg_count)`
- 提取函数参数 (TensorType) → host pointers + sizes
- 分配设备内存,拷贝输入数据
- 遍历函数体,为每个 Call 生成 add_task()
- 生成顺序依赖关系 (简化版)
- func_id 使用全局映射表: `func_name_to_id_`
- **CoreType 推断**: 扫描函数体内所有 block 操作,推断核类型

**CoreType 推断逻辑**:
```cpp
// 操作到核类型映射
std::map<std::string, CoreType> op_to_core_type_ = {
    // Matrix operations → CUBE
    {"block.matmul", CoreType::CUBE},
    {"block.matmul_acc", CoreType::CUBE},

    // Vector operations → VECTOR
    {"block.add", CoreType::VECTOR},
    {"block.mul", CoreType::VECTOR},
    {"block.div", CoreType::VECTOR},
    {"block.sub", CoreType::VECTOR},
    {"block.adds", CoreType::VECTOR},
    {"block.muls", CoreType::VECTOR},
    {"block.divs", CoreType::VECTOR},
    {"block.subs", CoreType::VECTOR},
    {"block.exp", CoreType::VECTOR},
    {"block.sqrt", CoreType::VECTOR},
    {"block.row_max", CoreType::VECTOR},
    {"block.row_sum", CoreType::VECTOR},
    {"block.row_expand_add", CoreType::VECTOR},
    {"block.row_expand_sub", CoreType::VECTOR},
    {"block.row_expand_mul", CoreType::VECTOR},
    {"block.row_expand_div", CoreType::VECTOR},

    // Memory operations - 中立,不影响判断
    {"block.load", CoreType::UNKNOWN},  // 特殊标记
    {"block.store", CoreType::UNKNOWN},
};

// 分析函数确定核类型
CoreType InferCoreType(const FunctionPtr& func) {
    std::set<CoreType> core_types;

    // 遍历所有 Call 操作
    for (auto call : CollectCallOps(func->body_)) {
        std::string op_name = call->op_->name_;
        if (op_to_core_type_.count(op_name)) {
            CoreType ct = op_to_core_type_[op_name];
            if (ct != CoreType::UNKNOWN) {  // 忽略 load/store
                core_types.insert(ct);
            }
        }
    }

    // 检查一致性
    CHECK(core_types.size() <= 1)
        << "Function " << func->name_ << " contains mixed core types. "
        << "All operations must use the same core type.";

    // 默认返回 VECTOR
    return core_types.empty() ? CoreType::VECTOR : *core_types.begin();
}
```

**张量分类**:
- 输入张量: 出现在 Call 参数中,但不是返回值 → DeviceMalloc + CopyToDevice
- 输出张量: 作为 Call 的返回值 → DeviceMalloc + RecordTensorPair
- 中间张量: 仅在函数内部分配 → DeviceMalloc

**func_id 映射表**:
```cpp
// 在生成时构建映射表
std::map<std::string, int> func_name_to_id_;
int next_func_id_ = 0;

int GetFuncId(const std::string& func_name) {
    if (!func_name_to_id_.count(func_name)) {
        func_name_to_id_[func_name] = next_func_id_++;
    }
    return func_name_to_id_[func_name];
}
```

**生成的代码包含映射注释**:
```cpp
// Function ID mapping:
//   0: rowmax
//   1: rowexpandsub
//   2: elem_exp
//   3: rowsum
//   4: rowexpanddiv
```

**阶段 2: 精确依赖分析 (后续)**
- 实现数据流依赖跟踪
- 生成最小依赖边
- 优化并行性

**阶段 3: 控制流支持 (后续)**
- 支持 ForStmt 循环
- 支持 IfStmt 条件分支

## 关键文件

**需要修改**:
- `src/codegen/pto_codegen.cpp` - 实现 GenerateOrchestrationCode()
- `include/pypto/codegen/pto_codegen.h` - 可能需要添加辅助方法声明

**参考文件**:
- `examples/ir_builder/dynamic_softmax_codegen.py` - 输入示例
- `/data/z00934994/zhusy54/simpler/example/kernels/orchestration/example_orch.cpp` - 输出示例
- `src/codegen/pto_codegen.cpp:809-824` - IsOrchestrationFunction 实现
- `/data/z00934994/zhusy54/simpler/src/runtime/host_build_graph/runtime/runtime.h` - Runtime API

## 测试验证

**单元测试** (`tests/ut/ir/transforms/test_pto_codegen.py`):
```python
def test_pto_codegen_orchestration():
    """Test orchestration code generation."""
    ib = IRBuilder()

    # Build simple orchestration function
    with ib.function("simple_orch") as f:
        input_t = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_t = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Call a kernel
        call_result = ir.Call(ir.GlobalVar("kernel_func"),
                              [input_t, output_t],
                              ir.Span.unknown())
        ib.let("result", call_result)
        ib.return_stmt(output_t)

    func = f.get_result()
    program = ir.Program([func], "test_orch", ir.Span.unknown())

    # Generate code
    codegen = ir.PTOCodegen()
    code = codegen.generate(program)

    # Verify output
    assert "int BuildSimple_orch(" in code
    assert "Runtime* runtime" in code
    assert "runtime->add_task(" in code
```

**集成测试**:
- 使用 `dynamic_softmax_codegen.py` 生成代码
- 检查生成的 C++ 代码是否可编译
- 验证任务图结构的正确性

## 代码质量要求

遵循 `.claude/rules/` 中的项目规则:

1. **错误检查** (`error-checking.md`):
   - 使用 CHECK 验证用户输入 (函数参数)
   - 使用 INTERNAL_CHECK 验证内部不变量
   - 抛出 PyPTO 异常,不使用 C++ 标准异常

2. **核心开发** (`core-development.md`):
   - 清晰的变量命名
   - 小而专注的函数
   - 使用现代 C++17 特性
   - 提供有帮助的错误消息

3. **文档** (`documentation.md`):
   - 添加函数注释说明生成逻辑
   - 更新相关文档(如有必要)

## MVP 实现清单

### 核心函数实现

**PTOCodegen 类新增方法**:
1. `CoreType InferFunctionCoreType(const FunctionPtr& func)` - 推断函数核类型
2. `int GetOrCreateFuncId(const std::string& func_name)` - 获取/创建 func_id
3. `FunctionPtr FindFunctionByName(const ProgramPtr& program, const std::string& name)` - 查找函数
4. `void GenerateOrchestrationCode(const FunctionPtr& func)` - 主实现函数

**PTOCodegen 类新增成员变量**:
```cpp
// 函数名到 ID 映射
std::map<std::string, int> func_name_to_id_;
int next_func_id_ = 0;

// 操作到核类型映射
static const std::map<std::string, std::optional<CoreType>> op_to_core_type_;

// 当前处理的 Program (需要查找函数定义)
const ProgramPtr* current_program_ = nullptr;
```

### 实现步骤概要

1. **函数签名生成** (30 行)
   - 生成 `extern "C"` 和函数签名
   - 参数验证: `if (arg_count < expected) return -1;`

2. **参数提取** (20 行)
   - 遍历 TensorType 参数
   - 生成 `void* host_X = ...` 和 `size_t size_X = ...`

3. **CoreType 推断** (60 行)
   - 实现 `InferFunctionCoreType()` 函数
   - 使用 IRVisitor 收集 block 操作
   - 验证核类型一致性

4. **设备内存分配** (40 行)
   - 分析输入/输出张量
   - 生成 DeviceMalloc、CopyToDevice、RecordTensorPair

5. **任务图生成** (80 行)
   - 遍历函数体收集 Call 操作
   - 为每个 Call 生成 add_task()
   - 生成顺序依赖: add_successor(t[i-1], t[i])

6. **辅助代码** (20 行)
   - 生成 func_id 映射注释
   - 生成统计和调试输出
   - 函数返回: `return 0;`

**总计**: 约 250 行代码

## 测试与验证

### 单元测试

**测试文件**: `tests/ut/ir/transforms/test_pto_codegen.py`

新增测试:
```python
def test_pto_codegen_orchestration_basic():
    """Test basic orchestration code generation."""
    ib = IRBuilder()

    # Build InCore function
    with ib.function("kernel_add") as f1:
        x = f1.param("x", ir.TileType([8, 8], DataType.FP32))
        y = f1.param("y", ir.TileType([8, 8], DataType.FP32))
        f1.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib.let("result", ir.op.block.add(x, y))
        ib.return_stmt(result)
    kernel_func = f1.get_result()

    # Build orchestration function
    with ib.function("simple_orch") as f2:
        a = f2.param("a", ir.TensorType([128], DataType.FP32))
        b = f2.param("b", ir.TensorType([128], DataType.FP32))
        f2.return_type(ir.TensorType([128], DataType.FP32))

        # Call kernel
        add_op = ir.GlobalVar("kernel_add")
        c = ib.let("c", ir.Call(add_op, [a, b], ir.Span.unknown()))
        ib.return_stmt(c)
    orch_func = f2.get_result()

    program = ir.Program([kernel_func, orch_func], "test_orch", ir.Span.unknown())

    # Generate code
    codegen = ir.PTOCodegen()
    code = codegen.generate(program)

    # Verify output
    assert "int BuildSimple_orch(" in code or "int BuildSimpleOrch(" in code
    assert "Runtime* runtime" in code
    assert "uint64_t* args" in code
    assert "runtime->add_task(" in code
    assert "DeviceMalloc" in code
    assert "CopyToDevice" in code or "RecordTensorPair" in code


def test_pto_codegen_core_type_inference():
    """Test core type inference from block operations."""
    ib = IRBuilder()

    # Vector function
    with ib.function("vector_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib.let("result", ir.op.block.mul(x, x))  # Vector op
        ib.return_stmt(result)
    vector_func = f.get_result()

    # Matrix function
    with ib.function("matrix_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        y = f.param("y", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib.let("result", ir.op.block.matmul(x, y))  # Matrix op
        ib.return_stmt(result)
    matrix_func = f.get_result()

    # Test inference
    codegen = ir.PTOCodegen()
    vector_core = codegen.infer_function_core_type(vector_func)
    matrix_core = codegen.infer_function_core_type(matrix_func)

    assert vector_core == ir.CoreType.VECTOR
    assert matrix_core == ir.CoreType.CUBE
```

### 集成测试

**测试脚本**: `test_simple_orchestration.py` (已存在)

运行验证:
```bash
python test_simple_orchestration.py
```

**预期输出**:
```cpp
extern "C" {

int BuildSimpleOrch(Runtime* runtime, uint64_t* args, int arg_count) {
    // Validate argument count
    if (arg_count < 6) {
        std::cerr << "Error: Expected at least 6 args" << std::endl;
        return -1;
    }

    // Extract arguments
    void* host_a = reinterpret_cast<void*>(args[0]);
    size_t size_a = static_cast<size_t>(args[1]);
    void* host_b = reinterpret_cast<void*>(args[2]);
    size_t size_b = static_cast<size_t>(args[3]);
    void* host_d = reinterpret_cast<void*>(args[4]);
    size_t size_d = static_cast<size_t>(args[5]);

    // Allocate device memory
    void* dev_a = runtime->host_api.DeviceMalloc(size_a);
    runtime->host_api.CopyToDevice(dev_a, host_a, size_a);

    void* dev_b = runtime->host_api.DeviceMalloc(size_b);
    runtime->host_api.CopyToDevice(dev_b, host_b, size_b);

    void* dev_d = runtime->host_api.DeviceMalloc(size_d);
    runtime->RecordTensorPair(host_d, dev_d, size_d);

    // Function ID mapping:
    //   0: kernel_add
    //   1: kernel_mul

    // Task 0: Call kernel_add
    uint64_t args_t0[2];
    args_t0[0] = reinterpret_cast<uint64_t>(dev_a);
    args_t0[1] = reinterpret_cast<uint64_t>(dev_b);
    int t0 = runtime->add_task(args_t0, 2, 0, 0);  // CoreType::VECTOR = 0

    // Task 1: Call kernel_mul
    uint64_t args_t1[2];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_c);
    args_t1[1] = reinterpret_cast<uint64_t>(dev_a);
    int t1 = runtime->add_task(args_t1, 2, 1, 0);  // CoreType::VECTOR = 0

    // Dependencies
    runtime->add_successor(t0, t1);

    return 0;
}

}  // extern "C"
```

### 编译验证

生成的 C++ 代码需要能够编译:
```bash
g++ -c generated_orch.cpp -I/path/to/simpler/include -std=c++17
```

验证点:
- 无编译错误
- 无未定义符号
- Runtime API 调用正确

### 功能验证

使用 `dynamic_softmax_codegen.py` 测试完整流程:
```bash
python examples/ir_builder/dynamic_softmax_codegen.py
```

检查生成的编排代码:
- 正确识别 5 个 InCore 函数 (rowmax, rowexpandsub, elem_exp, rowsum, rowexpanddiv)
- 正确分配设备内存 (6 个张量)
- 正确生成任务图
- 正确推断所有函数为 VECTOR 类型

## 已知限制 (MVP)

1. **依赖分析**: 仅生成顺序依赖 (t0→t1→t2...)
   - 后续改进: 数据流分析,生成最小依赖边

2. **控制流**: ForStmt 和 IfStmt 仅生成注释
   - 后续改进: 循环展开,条件分支处理

3. **中间张量**: Call 的返回值需要设备内存,但当前未处理
   - 临时方案: 假设所有中间结果已分配
   - 后续改进: 自动推断并分配中间张量内存

4. **张量大小计算**: 当前假设用户提供 size 参数
   - 后续改进: 从 TensorType shape 自动计算字节大小

## 风险与缓解

**风险 1**: CoreType 枚举值不匹配
- PyPTO: VECTOR=0, CUBE=1
- Simpler Runtime: AIC=0, AIV=1
- 缓解: 生成代码时添加注释说明枚举值含义

**风险 2**: func_id 冲突
- 缓解: 使用全局映射表,确保唯一性
- 生成映射表注释供用户检查

**风险 3**: 中间张量内存未分配
- 缓解: MVP 阶段假设用户已分配
- 添加 TODO 注释提醒后续改进

**风险 4**: 复杂控制流不支持
- 缓解: 生成注释说明需要手动处理
- 记录为后续改进项

## 后续改进 (Phase 2 & 3)

### Phase 2: 精确依赖分析

**目标**: 生成最小依赖边,提高并行性

**方法**:
- 跟踪每个变量的 producer 任务
- 跟踪每个任务的 input 变量
- 仅当 task B 使用 task A 的输出时,添加 A→B 依赖

### Phase 3: 控制流支持

**ForStmt 支持**:
- 固定次数循环: 展开生成多个任务
- 动态次数循环: 生成运行时循环代码

**IfStmt 支持**:
- 编译时常量条件: 仅生成一个分支
- 运行时条件: 生成条件任务调度代码

## 总结

MVP 实现将提供:
✅ 基本的编排代码生成框架
✅ CoreType 自动推断与验证
✅ func_id 映射表管理
✅ 输入/输出张量正确分类
✅ 顺序任务图生成
✅ 可编译的 C++ 代码输出

为后续改进奠定坚实基础。
