# `sbs-exercise` 作者使用指南

`sbs-exercise` 用于在书籍中插入 Python 代码练习。作者负责提供题目说明、初始代码和验证条件；用户将题目发送到代码栏、完成指定区域并运行后，平台自动给出正确或错误的判定。

这份指南面向会写普通 Python、但不熟悉单元测试的教材作者。出题时先确定“希望平台如何判断用户的答案”，再选择 `judgeMode`。

## 一、通用参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `title` | 是 | 题目标题 |
| `description` | 否 | 题目要求和作答说明 |
| `kind` | 是 | 作答类型：`completion` 或 `fix` |
| `language` | 否 | 代码语言；当前使用 `python`，省略时默认为 `python` |
| `code` | 是 | 发送到代码栏的初始代码 |
| `judgeMode` | 是 | 验证方式：`runSuccess`、`testCases` 或 `outputMatch` |
| `judgeTests` | 条件必填 | `judgeMode: testCases` 时必填，表示实际执行的测试用例 |
| `judgeOutput` | 条件必填 | `judgeMode: outputMatch` 时必填，表示作者规定的完整输出 |

请始终显式填写 `judgeMode`。如果遗漏或拼写错误，当前兼容逻辑可能将其当成 `runSuccess`，从而把“代码没有报错”误当成“答案正确”。

## 二、作答类型 `kind`

### `completion`

用于代码补全。作者在 `code` 中提供完整上下文，并使用标记指定用户需要补全的区域：

```python
# BEGIN_SOLUTION
pass
# END_SOLUTION
```

用户只能修改两个标记之间的内容。

### `fix`

用于代码改错。作者在指定区域内提供一段有问题的代码，用户需要在该区域中修复。

```python
# BEGIN_SOLUTION
return price * rate
# END_SOLUTION
```

`completion` 和 `fix` 使用相同的区域约束。

## 三、验证方式 `judgeMode`

### 1. `runSuccess`

适合以下效果：

- 用户补全后，整段脚本能够正常启动并结束。
- 代码没有唯一答案，作者只关心执行期间是否出现异常。

平台按普通脚本语义执行用户完成后的整段代码。作者可以正常书写：

```python
if __name__ == "__main__":
    main()
```

运行时平台会触发这类入口代码。代码能够正常运行结束且没有未捕获异常，即判定正确。

**必须参数：**

- `title`
- `kind`
- `judgeMode: runSuccess`
- `code`

**可选参数：**

- `language`
- `description`

```sbs-exercise
title: 生成欢迎语
kind: completion
judgeMode: runSuccess
description: |
  补全 build_greeting，使程序能够正常输出欢迎语。
code: |
  def build_greeting(name):
      # BEGIN_SOLUTION
      pass
      # END_SOLUTION

  def main():
      message = build_greeting("Alice")
      if not isinstance(message, str):
          raise TypeError("build_greeting 必须返回字符串")
      print(message)

  if __name__ == "__main__":
      main()
```

**平台如何判定：** 正常结束且没有未捕获异常时判定正确；运行中抛出异常时判定错误。

**易错情况：**

- `pass` 是合法的 Python 语句，本身不会报错。如果固定代码没有真正调用待补全逻辑，初始答案也可能被判为正确。
- 只定义 `main()` 不会自动运行。需要在代码最后写 `if __name__ == "__main__": main()`。
- `input()`、死循环和长时间网络请求可能让程序一直等待。
- 函数返回值、配置值或工具选择有明确要求时，不要仅使用 `runSuccess`，应改用 `testCases`。

### 2. `testCases`

适合以下效果：

- 用户补全函数，平台用多组输入检查返回值。
- 用户实现类或 Agent 工具逻辑，平台检查选中了哪个工具、传递了什么参数、返回了什么对象。

平台先以模块方式加载用户代码，不进入 `main()`，再执行作者写在 `judgeTests` 中的测试用例。所有测试通过才判定正确。

**必须参数：**

- `title`
- `kind`
- `judgeMode: testCases`
- `code`
- `judgeTests`

**可选参数：**

- `language`
- `description`

作者可以使用以下两种测试语句：

```python
check("测试名称", 实际结果（进行函数调用）, 期望结果)
check_true("测试名称", 条件表达式)
```

- `check`：比较实际结果和期望结果是否相等。
- `check_true`：检查类型、范围、结构等条件是否成立。
- `judgeTests`：运行时真正执行的完整测试。

测试名称应说明正在检查什么，例如“负数相加”“正确解析 JSON 参数”，不要只写“测试 1”。

如果需要向用户公开测试样例，作者可以直接将样例写在 `description` 中。

```sbs-exercise
title: 实现加法函数
kind: completion
judgeMode: testCases
description: |
  补全 add，使其返回两个参数之和。
  公开样例：add(1, 2) 应返回 3。
code: |
  def add(a, b):
      # BEGIN_SOLUTION
      pass
      # END_SOLUTION
judgeTests: |
  check("正数相加", add(1, 2), 3)
  check("负数相加", add(-1, -2), -3)
  check("与零相加", add(5, 0), 5)
  check_true("返回整数", isinstance(add(1, 2), int))
```

这组测试包含正常输入、一个简单边界和返回类型。对于这类简单题，不需要再加更复杂的测试。

#### Agent 工具分发示例

对于工具调用、LLM 客户端或外部 API，测试不要真正访问网络。作者可以写一个很小的假工具，记录它收到的参数。

```yaml
judgeMode: testCases
judgeTests: |
  class RecordingTool(BaseTool):
      name = "recording_tool"
      function_name = "recording_tool"

      def __init__(self):
          self.received_arguments = None

      def run(self, arguments):
          self.received_arguments = arguments
          return ToolResult(
              self.name,
              self.function_name,
              "执行成功",
              True,
              arguments,
          )

  recording_tool = RecordingTool()
  agent = TravelPlanAgent(
      llm_client=object(),
      tools=[recording_tool],
  )

  result = agent.execute_tool_call({
      "id": "call_001",
      "type": "function",
      "function": {
          "name": "recording_tool",
          "arguments": "{\"city\": \"杭州\", \"top_k\": 2}",
      },
  })

  check_true("返回 ToolResult", isinstance(result, ToolResult))

  if isinstance(result, ToolResult):
      check("调用正确工具", result.function_name, "recording_tool")
      check("返回工具结果", result.content, "执行成功")
      check("结果标记成功", result.success, True)

  check(
      "正确解析 JSON 参数",
      recording_tool.received_arguments,
      {"city": "杭州", "top_k": 2},
  )
```

这组测试只检查答案的可观察行为：找到正确工具、解析并传递参数、返回工具执行结果。它不要求答题者使用某个固定变量名或某种固定写法。

#### `judgeTests` 应该包含什么

作者不需要每道题都写很多测试。根据题目选择必要项即可：

| 检查内容 | 何时需要 | 示例 |
| --- | --- | --- |
| 基本正确性 | 每道题 | 正常输入得到期望结果 |
| 一个关键边界 | 存在空值、零、负数等边界时 | `add(5, 0)` 返回 `5` |
| 返回契约 | 要求特定类型或字段时 | 返回 `ToolResult` 且 `success=True` |
| 关键交互 | 要求调用工具或传递参数时 | 假工具收到解析后的 JSON 参数 |

一般使用 2–4 个有明确意义的检查就足够。

**易错情况：**

- 只测试“能调用”，却没有比较结果。
- 直接调用真实 LLM、天气、搜索或其他网络 API，导致相同答案因外部服务变化而得到不同结果。
- 检查答题者的变量名、循环次数或某种固定实现，而不是最终行为。
- 初始 `pass` 返回 `None` 时立即访问对象属性。应先用 `isinstance` 检查，再访问属性，以便给出清楚的失败结果。
- 只用正确答案试跑测试。作者还应确认未修改的 `pass` 会失败。

没有写任何 `check` / `check_true`，或用户代码、测试代码抛出异常，平台都会判定错误。

### 3. `outputMatch`

适合以下效果：

- 用户补全计算逻辑，程序最后应输出一个确定数字。
- 用户生成固定格式的单行或多行文本。

平台按普通脚本语义执行用户完成后的整段代码，捕获代码产生的全部标准输出，再与 `judgeOutput` 比较。两者一致时判定正确。

作者可以直接在 `code` 中写好函数调用或 `print` 语句，也可以使用 `main()` 加入口判断。只要整段脚本运行后产生的完整输出与 `judgeOutput` 一致即可。

**必须参数：**

- `title`
- `kind`
- `judgeMode: outputMatch`
- `code`
- `judgeOutput`

**可选参数：**

- `language`
- `description`

```sbs-exercise
title: 输出偶数之和
kind: completion
judgeMode: outputMatch
description: |
  补全 calculate_total，计算 1 到 10 中所有偶数的和。
code: |
  def calculate_total():
      # BEGIN_SOLUTION
      pass
      # END_SOLUTION

  def main():
      print(calculate_total())

  if __name__ == "__main__":
      main()
judgeOutput: |
  30
```

`judgeOutput` 可以包含多行内容。平台会将程序的完整输出与其进行整体比较：

```yaml
judgeOutput: |
  first line
  second line
```

平台会统一 Windows/Linux 换行符并忽略末尾多余换行，但行内空格、标点、大小写和额外输出仍会影响判定。

**易错情况：**

- 在标准结果之外打印调试信息，例如 `print("计算开始")`；额外文本也会参与比较。
- 输出包含当前时间、随机数、网络返回内容等不确定数据。
- 忘记调用 `main()`，导致脚本没有任何输出。
- 题目允许多种合理输出格式，却使用了严格的 `outputMatch`；此时更适合用 `testCases` 检查结构或关键字段。

## 四、作答区域约束

`completion` 和 `fix` 都必须在 `code` 中保留一组以下标记：

```python
# BEGIN_SOLUTION
# 用户允许修改的代码
# END_SOLUTION
```

## 五、选择建议

| 需求 | 推荐配置 |
| --- | --- |
| 代码能够正常运行即可 | `judgeMode: runSuccess` |
| 使用多组输入验证函数行为 | `judgeMode: testCases` |
| 比较程序最终打印的完整结果 | `judgeMode: outputMatch` |
| 补全缺失代码 | `kind: completion` |
| 修改指定区域内的错误代码 | `kind: fix` |

补充说明：`runSuccess` 和 `outputMatch` 按脚本运行，适合作者书写完整可运行程序；`testCases` 按模块加载，适合作者检查用户定义的函数、类或变量。带交互输入的 `outputMatch` 需要保证输入和输出可复现，否则不适合作为稳定自动判题。

只要题目存在明确的正确行为，例如“返回指定值”“调用正确工具”“解析并传递参数”，优先使用 `testCases`。

## 六、作者出题流程

1. 用一句话写清楚“用户需要完成什么”。
2. 将允许用户修改的内容放在唯一一组解答标记之间。
3. 根据判定目标选择 `runSuccess`、`testCases` 或 `outputMatch`。
4. 保留作者自己的正确答案，但不要写入用户看到的 `code`。
5. 先用未修改的初始代码运行，确认它会判定错误。
6. 再填入正确答案运行，确认它会判定正确。
7. 至少尝试一个常见错误答案，确认测试不会误放行。

## 七、发布前自检

- [ ] `judgeMode` 已显式填写且拼写正确。
- [ ] `code` 中有且只有一组解答标记。
- [ ] 用户只需要修改标记之间的代码。
- [ ] 初始 `pass` 或错误代码不会被判定正确。
- [ ] 正确答案能通过判定。
- [ ] `testCases` 不访问真实外部服务，测试结果可重复。
- [ ] `outputMatch` 的空格、换行、标点和大小写与实际输出一致。
- [ ] `runSuccess` / `outputMatch` 的完整脚本有可达的执行入口，不会无限等待。
