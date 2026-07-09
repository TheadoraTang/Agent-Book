# `sbs-exercise` 作者使用指南

`sbs-exercise` 用于在书籍中插入代码练习。作者负责提供题目说明、初始代码和验证条件；用户将题目发送到代码栏、完成指定区域并运行后，平台自动给出正确或错误的判定。

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

平台执行用户完成后的整段代码。代码能够正常运行结束且没有抛出异常，即判定正确。

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

  message = build_greeting("Alice")
  if not isinstance(message, str):
      raise TypeError("build_greeting 必须返回字符串")
  print(message)
```

### 2. `testCases`

平台先执行用户代码，再自动执行作者写在 `judgeTests` 中的测试用例。所有测试通过才判定正确。

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
  check_true("返回整数", isinstance(add(1, 2), int))
```

### 3. `outputMatch`

平台执行用户完成后的整段代码，捕获代码产生的全部标准输出，再与 `judgeOutput` 比较。两者一致时判定正确。

作者应直接在 `code` 中写好函数调用或 `print` 语句，确保整段代码运行后能够产生需要比较的输出。

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

  print(calculate_total())
judgeOutput: |
  30
```

`judgeOutput` 可以包含多行内容。平台会将程序的完整输出与其进行整体比较：

```yaml
judgeOutput: |
  first line
  second line
```

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
