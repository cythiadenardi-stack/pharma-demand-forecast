# 预测结果文件说明书 (Results Guide)

> 本文档面向**前端/可视化开发人员**，说明 `results/` 目录下所有预测结果文件的格式和含义。
> 
> 运行模型后自动生成这些文件，前端直接读取即可展示。

---

## 文件总览

运行完4个模型脚本后，`results/` 目录下会有以下文件：

| 文件 | 来源模型 | 内容 | 行数（约） |
|------|----------|------|-----------|
| `ets_predictions.csv` | ETS | 39 fast SKU × 4渠道 × 30天预测 | 4,680 |
| `ets_metrics.csv` | ETS | 39 SKU × 4渠道评估指标 | 156 |
| `prophet_predictions.csv` | Prophet/SARIMAX | 62 SKU × 4渠道 × 30天预测 | 7,440 |
| `prophet_metrics.csv` | Prophet/SARIMAX | 62 SKU × 4渠道评估指标 | 248 |
| `croston_predictions.csv` | Croston | 399 long_tail SKU × 4渠道 × 30天预测 | 47,880 |
| `croston_metrics.csv` | Croston | 399 SKU × 4渠道评估指标 | 1,596 |
| `xgboost_predictions.csv` | XGBoost | **500 SKU × 4渠道 × 30天预测** | **60,000** |
| `xgboost_metrics.csv` | XGBoost | 500 SKU × 4渠道评估指标 | 2,000 |
| `xgboost_feature_importance.csv` | XGBoost | 特征重要性排序 | ~30 |

> 注：XGBoost是"主力模型"，覆盖全部500个SKU，建议前端**主要展示XGBoost的结果**。

---

## 1. xgboost_predictions.csv — 主力预测结果（最重要的文件）

**用途**：展示每个SKU未来30天在每个渠道的预测销量

### 文件格式

```csv
sku_id,date,forecast_total,forecast_hospital,forecast_chain,forecast_independent
SKU-0001,2026-06-01,185.3,65.2,78.1,42.0
SKU-0001,2026-06-02,192.7,67.8,79.5,45.4
SKU-0001,2026-06-03,178.2,62.1,74.3,41.8
...
SKU-0002,2026-06-01,12.5,4.1,5.2,3.2
```

### 字段说明

| 字段 | 类型 | 示例 | 含义 |
|------|------|------|------|
| `sku_id` | 字符串 | SKU-0001 | 药品编码 |
| `date` | 日期 | 2026-06-01 | 预测日期（2026-06-01 ~ 2026-06-30） |
| `forecast_total` | 浮点数 | 185.3 | **全国总销量预测**（盒/天） |
| `forecast_hospital` | 浮点数 | 65.2 | **医院渠道**预测销量 |
| `forecast_chain` | 浮点数 | 78.1 | **连锁药店**预测销量 |
| `forecast_independent` | 浮点数 | 42.0 | **独立药店**预测销量 |

### 关键特性

- **每天预测值不同**：XGBoost用递推预测，30天每天都不一样
- **渠道可加**：forecast_total ≈ forecast_hospital + forecast_chain + forecast_independent
- **有500个SKU**，每个SKU有30天 × 4渠道 = 120个预测值

### 前端怎么用这个文件

```python
import pandas as pd

# 读取
df = pd.read_csv("results/xgboost_predictions.csv")

# 选一个SKU展示
df_sku = df[df['sku_id'] == 'SKU-0001']

# 画折线图：日期 vs forecast_total（总销量趋势）
# 画堆叠面积图：日期 vs [hospital, chain, independent]（渠道拆分）
```

---

## 2. xgboost_metrics.csv — 模型准确度评估

**用途**：告诉用户"这个预测有多准"

### 文件格式

```csv
sku_id,channel,mape,rmse
SKU-0001,total,12.5,98.3
SKU-0001,hospital,15.2,42.1
SKU-0001,chain,14.8,38.5
SKU-0001,independent,18.3,28.7
SKU-0002,total,8.3,15.2
```

### 字段说明

| 字段 | 类型 | 示例 | 含义 |
|------|------|------|------|
| `sku_id` | 字符串 | SKU-0001 | 药品编码 |
| `channel` | 字符串 | total | 渠道（total/hospital/chain/independent） |
| `mape` | 浮点数 | 12.5 | **平均绝对百分比误差**（%），越小越准 |
| `rmse` | 浮点数 | 98.3 | **均方根误差**（盒），越小越准 |

### MAPE怎么解读

| MAPE范围 | 准确度 | 前端展示建议 |
|----------|--------|-------------|
| < 10% | 非常准 | 绿色 ✅ |
| 10-20% | 比较准 | 蓝色 ℹ️ |
| 20-30% | 一般 | 黄色 ⚠️ |
| > 30% | 不太准 | 红色 ❌ |

### 前端怎么用这个文件

```python
# 显示"模型准确度: MAPE=12.5%"这样的文字
# 用颜色条/进度条展示准确度
# 按SKU排序找出"最难预测"的药品
```

---

## 3. xgboost_feature_importance.csv — 特征重要性

**用途**：解释"为什么模型这样预测"（AI可解释性）

### 文件格式

```csv
feature,importance
sales_qty_lag_7,0.285
flu_activity_index,0.192
sales_rolling_mean_7,0.156
is_holiday,0.087
month_sin,0.065
...
```

### 字段说明

| 字段 | 类型 | 示例 | 含义 |
|------|------|------|------|
| `feature` | 字符串 | sales_qty_lag_7 | 特征名称 |
| `importance` | 浮点数 | 0.285 | 重要性权重（0~1，总和=1） |

### Top特征解读

| 特征名 | 中文 | 含义 |
|--------|------|------|
| `sales_qty_lag_7` | 7天前销量 | 上周同一天卖了多少 |
| `sales_rolling_mean_7` | 过去7天平均 | 近期平均走势 |
| `flu_activity_index` | 流感活动指数 | 流感季对药品需求的影响 |
| `is_holiday` | 是否假期 | 假期药店关门 |
| `month_sin/month_cos` | 月份编码 | 季节性周期 |
| `is_post_vbp` | 集采后标记 | 集采降价带来的需求跃升 |
| `lag_14/lag_30` | 14天/30天前 | 更长期的滞后效应 |

### 前端怎么用这个文件

```python
# 画水平条形图：feature vs importance
# 标题："影响销量预测的关键因素"
# 给用户解释："这个预测主要基于上周销量和流感指数"
```

---

## 4. 其他模型预测结果（对比基线用）

### 4.1 ets_predictions.csv — ETS指数平滑预测

**格式与xgboost_predictions.csv相同**，但只覆盖39个fast SKU。

**用途**：和XGBoost对比，展示"传统方法 vs 机器学习方法"的差异。

### 4.2 prophet_predictions.csv — Prophet季节性分解预测

**格式与xgboost_predictions.csv相同**，覆盖62个seasonal+policy_shocked SKU。

**特殊字段**：prophet会额外输出 `trend`（趋势分量）、`seasonal`（季节分量），可以画分解图。

### 4.3 croston_predictions.csv — Croston间歇需求预测

**格式稍有不同**：

```csv
sku_id,date,channel,croston_forecast,sba_forecast
SKU-0002,2026-06-01,total,10.7894,10.2500
```

| 字段 | 说明 |
|------|------|
| `croston_forecast` | 原始Croston预测 |
| `sba_forecast` | SBA改进版预测（**推荐用这个**，更准确） |

**特性**：Croston是"点预测"，30天预测值相同（这是理论特性，不是bug）。

**前端建议**：对long_tail SKU画柱状图而不是折线图（因为每天都是同一个预测值）。

---

## 5. croston_metrics.csv / ets_metrics.csv / prophet_metrics.csv

格式与 `xgboost_metrics.csv` 相同。

**croston_metrics.csv 特殊字段**：

```csv
sku_id,channel,nonzero_ratio,croston_forecast,sba_forecast,rmse_croston,rmse_sba,mae_croston,mae_sba,mape_croston,mape_sba
```

| 额外字段 | 说明 |
----------|------|
| `nonzero_ratio` | 历史数据中 nonzero 的比例（判断"有多间歇"） |
| `rmse_croston` / `rmse_sba` | 两种方法的RMSE对比 |
| `mae_croston` / `mae_sba` | 两种方法的MAE对比 |

**前端展示**：对比Croston vs SBA的MAE，说明"SBA改进了X%"

---

## 6. 前端展示建议（快速参考）

### 6.1 核心展示：预测趋势图

**数据来源**：`xgboost_predictions.csv`

```
图表类型：折线图
X轴：date（2026-06-01 至 2026-06-30）
Y轴：forecast_total（或分渠道展示）
交互：下拉框选择SKU，切换显示不同药品
```

### 6.2 渠道拆分：堆叠面积图

**数据来源**：`xgboost_predictions.csv`

```
图表类型：堆叠面积图
X轴：date
Y轴：forecast_hospital + forecast_chain + forecast_independent
颜色：不同渠道不同颜色
```

### 6.3 准确度仪表盘

**数据来源**：`xgboost_metrics.csv`

```
展示内容：
- 平均MAPE（所有SKU的total渠道平均）
- MAPE分布直方图
- "最准预测TOP5"和"最难预测TOP5"SKU列表
```

### 6.4 特征重要性：水平条形图

**数据来源**：`xgboost_feature_importance.csv`

```
图表类型：水平条形图
X轴：importance
Y轴：feature（中文翻译后）
用途：向用户解释"AI是怎么做出这个预测的"
```

---

## 7. 数据字典：SKU相关文件

前端展示SKU信息时需要用到以下数据文件：

### products.csv — SKU基本信息

| 字段 | 用途 |
|------|------|
| `sku_id` | 唯一标识 |
| `sku_name` | 中文名称（如"氨氯地平片 5mgx28"） |
| `generic_name_cn` | 通用名（如"氨氯地平"） |
| `therapy_area` | 治疗领域（如"Cardiovascular"） |
| `vbp_flag` | 是否集采（0/1） |
| `unit_price_cny` | 单价（元） |

### sku_profiles.csv — SKU分类标签

| 字段 | 用途 |
|------|------|
| `sku_id` | 唯一标识 |
| `demand_class` | 需求分类（fast/seasonal/long_tail/policy_shocked） |
| `base_demand` | 基础日均需求 |

### replenishment.csv — 补货建议

| 字段 | 用途 |
|------|------|
| `sku_id` | 唯一标识 |
| `current_inventory` | 当前库存 |
| `suggested_order_qty` | **建议订货量** |
| `priority` | 优先级（High/Medium/Low） |

---

*本文档生成时间: 2026-06-05*
*对应数据生成: SEED=42*
