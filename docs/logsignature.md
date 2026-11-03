# Logsignature 计算详解

## 整体 pipeline

```
路径 [batch, N+1, 2]
  │
  ▼ Phase 1: Chen's identity
截断签名 S = [S₁, S₂, S₃]   维度 [2, 4, 8]
  │
  ▼ Phase 2: 张量对数
log(S) = [L₁, L₂, L₃]       维度 [2, 4, 8]
  │
  ▼ Phase 3: Lyndon 提取
logsignature = [5个坐标]      维度 [5]
```

以下用 **d=2（二维路径）, depth=3** 作为贯穿示例。路径的两个通道记为 x 和 y（对应 t 和 T_t）。

---

## 数据结构：截断张量代数

所有计算都在**截断张量代数** T³(ℝ²) 中进行。一个元素是一组张量：

```
Level 0:  标量 1（隐含，不显式存储）
Level 1:  ℝ² 中的向量          → 存为 [batch, 2]
Level 2:  ℝ²⊗ℝ² = ℝ²ˣ² 中的矩阵  → 展平存为 [batch, 4]
Level 3:  ℝ²⊗ℝ²⊗ℝ² 中的 3阶张量  → 展平存为 [batch, 8]
```

展平规则：多重指标 `(i₁, i₂, ..., iₖ)` 映射到一维索引 `i₁·dᵏ⁻¹ + i₂·dᵏ⁻² + ... + iₖ`。例如 d=2 时：

```
Level 2 的 4 个位置:
  (0,0)→0  (0,1)→1  (1,0)→2  (1,1)→3
  即 [S^{xx}, S^{xy}, S^{yx}, S^{yy}]

Level 3 的 8 个位置:
  (0,0,0)→0  (0,0,1)→1  (0,1,0)→2  (0,1,1)→3
  (1,0,0)→4  (1,0,1)→5  (1,1,0)→6  (1,1,1)→7
```

贯穿整个计算的两个核心操作：

### 张量积 ⊗

`_tensor_product_levels(left, right)` 做外积然后展平：

```python
# left: [batch, dʲ],  right: [batch, dᵐ]
# → [batch, dʲ, dᵐ] → [batch, dʲ⁺ᵐ]
(left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(batch, -1)
```

### 截断乘法

两个张量代数元素相乘，第 k 层 = Σⱼ (左边第 j 层 ⊗ 右边第 k-j 层)，截断到 depth。

---

## Phase 1: 签名计算（Chen's identity）

### 数学定义

路径 γ: [0,1] → ℝᵈ 的 **level-k 签名** 是 k 重迭代积分：

$$S^{i_1 i_2 \cdots i_k} = \int_0^1 \int_0^{t_k} \cdots \int_0^{t_2} d\gamma^{i_1}_{t_1} \, d\gamma^{i_2}_{t_2} \cdots d\gamma^{i_k}_{t_k}$$

对于分段线性路径（我们的情况），用 **Chen's identity** 把它分解成逐段计算。

### 算法

输入：路径点 `p₀, p₁, ..., pₙ`，增量 `Δxₙ = pₙ₊₁ - pₙ`

```
初始化: S = 恒元 (level 0 = 1, 其余 = 0)

对每个增量 Δx:
    1. 计算 exp(Δx) 在截断张量代数中
    2. S ← S · exp(Δx)
```

### 第一步：单段的 exp(Δx)

对一个线性段（增量 Δx ∈ ℝᵈ），它的签名就是张量指数：

$$\exp(\Delta x) = 1 + \Delta x + \frac{\Delta x^{\otimes 2}}{2!} + \frac{\Delta x^{\otimes 3}}{3!} + \cdots$$

代码逐级构造：

```
level 1:  Δx                           → [batch, 2]
level 2:  Δx ⊗ Δx / 2!                → [batch, 4]
level 3:  Δx ⊗ Δx ⊗ Δx / 3!          → [batch, 8]
```

具体来说，`power` 变量迭代地做 `power ← power ⊗ Δx`：

```python
power = inc                              # Δx,        shape [batch, 2]
exp_levels[0] = power / 1!               # Δx / 1
power = _tensor_product_levels(power, inc)  # Δx⊗Δx,   shape [batch, 4]
exp_levels[1] = power / 2!               # Δx⊗Δx / 2
power = _tensor_product_levels(power, inc)  # Δx⊗Δx⊗Δx, shape [batch, 8]
exp_levels[2] = power / 3!               # Δx⊗Δx⊗Δx / 6
```

### 第二步：Chen's identity 乘法

已有签名 S 和当前段 E = exp(Δx)，新签名 S' = S · E：

$$(S \cdot E)_k = \sum_{j=0}^{k} S_j \otimes E_{k-j}$$

其中 S₀ = E₀ = 1（标量恒元）。代码拆成三类项：

```python
for k in range(depth):           # k=0,1,2 → level 1,2,3
    level_k  = exp_levels[k]     # j=0 项: 1 ⊗ Eₖ₊₁ = Eₖ₊₁
    level_k += sig[k]            # j=k+1 项: Sₖ₊₁ ⊗ 1 = Sₖ₊₁
    for j in range(1, k+1):      # 中间项: Sⱼ ⊗ Eₖ₊₁₋ⱼ
        level_k += sig[j-1] ⊗ exp_levels[k-j]
```

### 具体例子：d=2, depth=3, 路径 (0,0)→(1,0)→(1,1)

```
Δx₁ = (1,0),  Δx₂ = (0,1)

─── 处理 Δx₁ = (1,0) ───
exp(Δx₁):
  level 1: [1, 0]
  level 2: [1,0,0,0] / 2 = [0.5, 0, 0, 0]      # (1,0)⊗(1,0)/2
  level 3: [1,0,0,0,0,0,0,0] / 6 = [0.167, ...]  # (1,0)⊗(1,0)⊗(1,0)/6

S = 0 初始, 所以 S ← exp(Δx₁):
  S₁ = [1, 0]
  S₂ = [0.5, 0, 0, 0]
  S₃ = [0.167, 0, 0, 0, 0, 0, 0, 0]

─── 处理 Δx₂ = (0,1) ───
exp(Δx₂):
  E₁ = [0, 1]
  E₂ = [0,0,0,0.5]     # (0,1)⊗(0,1)/2
  E₃ = [0,...,0.167]    # (0,1)⊗(0,1)⊗(0,1)/6

S' = S · exp(Δx₂), level by level:

  S'₁ = E₁ + S₁ = [0,1] + [1,0] = [1, 1]
       ↑               ↑
       1·E₁            S₁·1

  S'₂ = E₂ + S₂ + S₁⊗E₁
       = [0,0,0,0.5] + [0.5,0,0,0] + [1,0]⊗[0,1]
       = [0,0,0,0.5] + [0.5,0,0,0] + [0,1,0,0]
       = [0.5, 1, 0, 0.5]
         ^xx  ^xy ^yx ^yy

  S'₃ = E₃ + S₃ + S₁⊗E₂ + S₂⊗E₁
       = ... (8维向量，省略)
```

注意 `S'₂` 中的 `S^{xy} = 1` 而 `S^{yx} = 0`。它们的差 `S^{xy} - S^{yx} = 1` 就是路径围成的（有向）面积。

---

## Phase 2: 张量对数

### 数学定义

签名 S 是一个 group-like 元素（S₀ = 1）。它的对数定义为：

$$\log(S) = \log(1 + A) = A - \frac{A^2}{2} + \frac{A^3}{3} - \cdots$$

其中 A = S - 1（去掉 level 0 的恒元部分）。

关键性质：A 在 level 0 为零，所以 Aⁿ 在 level < n 为零。因此级数在 n = depth 处**精确截断**（不是近似！）。

### 算法

```python
A = [S₁, S₂, S₃]           # A = S - 1，level 0 已经是 0

result = A.clone()           # 初始化为 A（n=1 项）

A_power = A                  # A¹
for n = 2, 3:
    A_power = A_power * A    # 截断张量代数乘法
    coeff = (-1)^(n+1) / n   # +1/2, -1/3, ...
    result += coeff * A_power
```

展开：

```
log(S)₁ = A₁                                    = S₁
log(S)₂ = A₂ - (A·A)₂ / 2                      = S₂ - S₁⊗S₁ / 2
log(S)₃ = A₃ - (A·A)₃ / 2 + (A·A·A)₃ / 3
```

其中 `(A·A)₂ = A₁ ⊗ A₁`（因为 A₀ = 0，乘积 level 2 只有一个非零项）。

### 具体例子（续上）

```
S₁ = [1, 1],  S₂ = [0.5, 1, 0, 0.5]

log(S)₁ = S₁ = [1, 1]

log(S)₂ = S₂ - S₁⊗S₁ / 2
         = [0.5, 1, 0, 0.5] - [1,1]⊗[1,1] / 2
         = [0.5, 1, 0, 0.5] - [1, 1, 1, 1] / 2
         = [0.5, 1, 0, 0.5] - [0.5, 0.5, 0.5, 0.5]
         = [0, 0.5, -0.5, 0]
           ^xx  ^xy   ^yx  ^yy
```

注意 `log(S)₂` 是**反对称**的：`L^{xy} = 0.5 = -L^{yx}`。这不是巧合——log(S) 是一个 Lie 代数元素，level 2 部分必然反对称。

---

## Phase 3: Lyndon 词提取

### 为什么需要这步

log(S) 在张量代数中有 2+4+8 = 14 个分量，但大部分是冗余的（因为 Lie 代数有对称约束）。独立的分量数 = logsignature 的维度 = 5。

### Lyndon 词

Lyndon 词是字母表 {0, 1} 上**严格小于自身所有真后缀**的字符串：

```
Level 1:  (0), (1)           — 2 个
Level 2:  (0,1)              — 1 个  （(1,0) 不是 Lyndon: 后缀 (0) < (1,0)）
Level 3:  (0,0,1), (0,1,1)  — 2 个
                              ─────
                              共 5 个
```

### 提取规则

**定理**：log(S) 在 Lyndon 基下的坐标，等于 log(S) 在张量代数中对应 Lyndon 词多重指标处的分量。

即：直接从 log(S) 的展平向量中按 Lyndon 词的索引位置**读取**即可。

```
Lyndon 词   →  多重指标    →  展平索引   →  从哪读
(0)         →  (0)         →  0          →  log(S)₁[0]
(1)         →  (1)         →  1          →  log(S)₁[1]
(0,1)       →  (0,1)       →  0·2+1 = 1  →  log(S)₂[1]
(0,0,1)     →  (0,0,1)     →  0·4+0·2+1 = 1  →  log(S)₃[1]
(0,1,1)     →  (0,1,1)     →  0·4+1·2+1 = 3  →  log(S)₃[3]
```

### 具体例子（续上）

```
logsig[0] = log(S)₁[0] = 1      ← x方向总增量
logsig[1] = log(S)₁[1] = 1      ← y方向总增量
logsig[2] = log(S)₂[1] = 0.5    ← Lévy 面积（路径围成的有向面积）
logsig[3] = log(S)₃[1] = ...    ← 三阶项
logsig[4] = log(S)₃[3] = ...    ← 三阶项
```

### 为什么这个提取是对的

以 level 2 为例。log(S)₂ 是一个 Lie 代数元素，可以写成：

$$L_2 = c_{01} \cdot [e_0, e_1]$$

其中 `[e₀, e₁] = e₀⊗e₁ - e₁⊗e₀` 展开在张量代数中是 `[0, 1, -1, 0]`（位置 xy=+1, yx=-1）。

所以：
- `L₂[xy位置] = c₀₁ · 1 = c₀₁`
- `L₂[yx位置] = c₀₁ · (-1) = -c₀₁`

读取 xy 位置（= Lyndon 词 (0,1) 的位置）就直接得到系数 `c₀₁`。

level 3 同理：两个 Lyndon 词 (0,0,1) 和 (0,1,1) 对应的 Lie bracket 在张量代数中展开后，各自在自己的 Lyndon 词位置系数为 1。所以直接读取即可。

---

## 各 logsignature 分量的物理含义

| 分量 | Lyndon 词 | 含义 |
|------|-----------|------|
| logsig[0] | (0) | x 通道（时间 t）的总增量 |
| logsig[1] | (1) | y 通道（时钟 T_t）的总增量 |
| logsig[2] | (0,1) | **Lévy 面积**：路径 (t, T_t) 围成的有向面积，编码了 t 和 T_t 之间的"交织"程度 |
| logsig[3] | (0,0,1) | 三阶项：捕捉 t 变化主导下 T_t 如何响应 |
| logsig[4] | (0,1,1) | 三阶项：捕捉 T_t 变化主导下的累积效应 |

---

## 维度公式

每一层的独立 Lyndon 词数由 **Witt 公式** 给出：

$$L(d, k) = \frac{1}{k} \sum_{m \mid k} \mu(k/m) \cdot d^m$$

其中 μ 是 Möbius 函数。logsignature 总维度 = Σₖ L(d, k)。

| depth | 签名维度 (d=2) | logsig 维度 (d=2) | 压缩比 |
|-------|:-:|:-:|:-:|
| 1 | 2 | 2 | 1× |
| 2 | 6 | 3 | 2× |
| 3 | 14 | 5 | 2.8× |
| 4 | 30 | 8 | 3.75× |
| 5 | 62 | 14 | 4.4× |

---

## 计算复杂度

时间复杂度：O(seq_len × depth² × d^depth)

对于默认参数 d=2, depth=3, seq_len=50：约 50 × 9 × 8 = 3600 次张量操作，开销很小。
