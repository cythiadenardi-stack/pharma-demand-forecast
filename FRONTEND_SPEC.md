# 前端展示规格说明书 (Frontend Specification)

> 本文档基于老师PLAN.md的要求，详细列出Streamlit前端应用需要展示的**所有页面、组件和数据来源**。
> 
> 面向：**前端/可视化开发人员（队友）**
> 技术栈：**Streamlit + Plotly**

---

## 应用总体架构

```
Streamlit多页面应用（Sidebar导航）
│
├── 页面1: 需求预测看板 (Demand Forecasting)
│   ├── SKU选择器
│   ├── 历史销量 vs 预测销量折线图
│   ├── 渠道拆分堆叠图
│   ├── 预测准确度指标
│   └── 特征重要性解释
│
├── 页面2: 智能补货建议 (Replenishment Recommendations)
│   ├── 库存状态总览
│   ├── 补货优先级清单
│   ├── 单个SKU补货详情
│   └── 建议订货量计算器
│
├── 页面3: 库存健康诊断 (Inventory Health)
│   ├── ABC/XYZ矩阵散点图
│   ├── 滞销品清单
│   ├── 近效期预警
│   └── 库存周转KPI
│
├── 页面4: 政策影响模拟 (Policy Simulator)
│   ├── VBP集采场景选择
│   ├── 集采前后对比
│   ├── 医院→零售需求转移模拟
│   └── 收入影响估算
│
└── 页面5: 预警通知 (Alerts)
    ├── 缺货预警列表
    └── 库存积压预警
```

---

## 页面1: 需求预测看板 (Demand Forecasting)

**核心功能**：选择任意SKU，查看其历史销量和30天预测趋势

### 1.1 SKU选择器

```
组件: 下拉选择框 (st.selectbox)
数据源: products.csv['sku_name'] 或 products.csv['generic_name_cn']
联动: 选择后更新下方所有图表
```

### 1.2 历史销量 vs 预测销量折线图（核心图表）

```python
图表类型: Plotly折线图

数据源: 
  - 历史销量: demand_daily.csv (2020-01-01 ~ 2026-05-31)
  - 预测销量: results/xgboost_predictions.csv (2026-06-01 ~ 2026-06-30)

X轴: date（日期）
Y轴: 销量（盒）

线条:
  - 蓝色实线: 历史实际销量 (demand_total)
  - 红色虚线: 未来30天预测 (forecast_total)
  - 灰色阴影: 置信区间（可选，用预测值±10%）

交互:
  - 鼠标悬停显示具体数值
  - 可缩放时间范围
  - 图例可点击隐藏/显示

展示时长: 默认展示最近6个月历史 + 30天预测
```

### 1.3 渠道拆分堆叠面积图

```python
图表类型: Plotly堆叠面积图

数据源: results/xgboost_predictions.csv

X轴: date
Y轴: 销量（盒）
堆叠层:
  - 浅蓝色: forecast_hospital (医院)
  - 浅绿色: forecast_chain (连锁药店)
  - 浅橙色: forecast_independent (独立药店)

总和: ≈ forecast_total

展示: 只展示预测期（2026-06-01 ~ 2026-06-30）
```

### 1.4 预测准确度指标卡

```python
组件: 3列指标卡 (st.columns(3))

数据源: results/xgboost_metrics.csv

卡片1: MAPE
  标题: "预测准确度"
  数值: "{mape}%"
  说明: "平均绝对百分比误差"
  颜色: <10%绿色, 10-20%蓝色, 20-30%黄色, >30%红色

卡片2: RMSE
  标题: "预测误差"
  数值: "{rmse} 盒"
  说明: "均方根误差"
  
卡片3: 趋势方向
  标题: "销量趋势"
  数值: "↗ 上升" / "↘ 下降" / "→ 平稳"
  说明: "基于预测期最后7天 vs 前7天"
```

### 1.5 特征重要性解释（AI可解释性）

```python
图表类型: Plotly水平条形图

数据源: results/xgboost_feature_importance.csv

Y轴: feature（中文翻译）
X轴: importance

Top特征翻译:
  sales_qty_lag_7 → "7天前销量"
  flu_activity_index → "流感活动指数"
  sales_rolling_mean_7 → "近7天平均销量"
  is_holiday → "是否假期"
  month_sin/month_cos → "季节性"
  is_post_vbp → "集采后"

标题: "影响该药品销量预测的关键因素"
副标题: "XGBoost模型告诉我们，这些因素影响最大"
```

---

## 页面2: 智能补货建议 (Replenishment Recommendations)

**核心功能**：基于预测结果，告诉用户"现在该订多少货"

### 2.1 库存状态总览

```python
组件: 3列大数字指标 (st.columns(3))

数据源: replenishment.csv

计算逻辑:
  high_count = (replenishment['priority'] == 'High').sum()
  medium_count = (replenishment['priority'] == 'Medium').sum()
  low_count = (replenishment['priority'] == 'Low').sum()

卡片1: 紧急补货
  数值: "{high_count} SKU"
  颜色: 红色背景
  说明: "库存低于安全线，可能马上断货"

卡片2: 需要补货
  数值: "{medium_count} SKU"
  颜色: 黄色背景
  说明: "库存低于再订货点"

卡片3: 库存充足
  数值: "{low_count} SKU"
  颜色: 绿色背景
  说明: "库存健康"
```

### 2.2 补货优先级清单（核心表格）

```python
组件: 数据表格 (st.dataframe) + 搜索/筛选

数据源: replenishment.csv + xgboost_predictions.csv

表格列:
  | 优先级 | SKU名称 | 当前库存 | 安全库存 | 可售天数 | 建议订货量 | 订货金额 | 操作 |

列说明:
  优先级: High=🔴, Medium=🟡, Low=🟢
  SKU名称: products.csv['sku_name']
  当前库存: replenishment['current_inventory']
  安全库存: replenishment['safety_stock']
  可售天数: current_inventory / avg_daily_demand
  建议订货量: replenishment['suggested_order_qty']
  订货金额: replenishment['order_value_cny']

排序: 按优先级（High→Medium→Low），同优先级按订货金额降序

筛选: 可按therapy_area（治疗领域）、priority（优先级）筛选
```

### 2.3 单个SKU补货详情

```python
点击表格行 → 展开详情面板

显示内容:
  1. SKU基本信息（名称、规格、供应商、价格）
  2. 当前库存 vs 安全库存的进度条
     [████████░░░░░░░░░░] 56% (库存健康)
  3. 未来14天预测销量折线图
     数据源: xgboost_predictions.csv 取前14天
  4. 建议订货量计算公式展示:
     "建议订货量 = (未来14天预测需求: 245盒) + (安全库存: 150盒) - (当前库存: 89盒) = 306盒"
  5. 预计到货日期（今天 + lead_time_days）
```

### 2.4 建议订货量计算器（可选互动功能）

```python
组件: 输入框 + 计算按钮

用户输入:
  - 当前库存: [   ] 盒
  - 日均销量: [   ] 盒
  - 安全天数: [14 ] 天（默认14天）
  - 配送天数: [ 7 ] 天（默认7天）

计算:
  建议订货量 = max(0, 日均销量 × (安全天数 + 配送天数) - 当前库存)

输出:
  "建议订货: XXX盒，预计花费: YYY元"
```

---

## 页面3: 库存健康诊断 (Inventory Health)

**核心功能**：分析库存结构，识别问题和机会

### 3.1 ABC/XYZ矩阵散点图（核心图表）

```python
图表类型: Plotly散点图

数据源: 
  - X轴（价值贡献）: 从demand_daily.csv计算每个SKU的月均销售额
  - Y轴（需求波动）: 从demand_daily.csv计算每个SKU销量的变异系数(CV)

ABC分类（X轴）:
  A类: 销售额Top 20%（累计80%销售额）→ 红色
  B类: 销售额Top 20-50% → 橙色
  C类: 销售额Bottom 50% → 灰色

XYZ分类（Y轴）:
  X类: CV < 0.5（需求稳定）→ 上部
  Y类: CV 0.5-1.0（中等波动）→ 中部
  Z类: CV > 1.0（高度波动）→ 下部

象限含义:
  AX: 高价值+稳定 → 重点管理，自动补货
  AZ: 高价值+波动 → 安全库存要高
  CX: 低价值+稳定 → 批量订货，减少频次
  CZ: 低价值+波动 → 考虑淘汰或合并

交互:
  - 鼠标悬停显示SKU名称和分类
  - 可缩放
  - 点击SKU跳转到预测页面
```

### 3.2 滞销品清单

```python
组件: 表格

数据源: demand_daily.csv + sku_profiles.csv

筛选逻辑:
  demand_class == 'long_tail' AND 近90天平均日销量 < 1盒

表格列:
  | SKU名称 | 治疗领域 | 90天总销量 | 当前库存 | 库存周转天数 | 建议 |

建议列内容:
  - 如果库存周转 > 180天: "建议促销清仓"
  - 如果库存周转 90-180天: "建议减少订货"
  - 如果库存周转 < 90天: "正常"
```

### 3.3 近效期预警

```python
组件: 警告卡片 + 表格

数据源: products.csv + inventory.csv

逻辑:
  products['shelf_life_months'] - 已存放月数 < 6个月

展示:
  🔴 警告: "XX个SKU库存将在6个月内过期！"
  
表格列:
  | SKU名称 | 入库日期 | 保质期(月) | 剩余有效期 | 库存量 | 风险等级 |
```

### 3.4 库存周转KPI

```python
组件: 4列指标卡

计算逻辑（从demand_daily.csv和inventory.csv）:
  
卡片1: 平均库存周转天数
  "45天"
  说明: "库存从进货到卖出平均需要45天"

卡片2: 缺货SKU数
  "12 SKU"
  说明: "过去30天发生过缺货的SKU数量"
  颜色: >10黄色, >20红色

卡片3: 库存总金额
  "¥2.3M"
  说明: "当前库存总价值"

卡片4: 预测准确率
  "87.3%"
  说明: "(1-MAPE) × 100"
  颜色: >90%绿色, 80-90%蓝色, <80%黄色
```

---

## 页面4: 政策影响模拟 (Policy Simulator)

**核心功能**：模拟集采等政策变化对药品需求的影响

### 4.1 VBP集采场景选择

```python
组件: 下拉选择框 + 参数滑块

场景预设（下拉选择）:
  - 第1批集采（2019-12，心血管/肿瘤）
  - 第4批集采（2021-05，广泛品种）
  - 第9批集采（2023-11，新批次）
  - 自定义场景

参数滑块:
  - 降价幅度: 0% ~ 80%（默认51%）
  - 涉及SKU数: 1 ~ 50
  - 模拟时长: 1 ~ 12个月
```

### 4.2 集采前后对比

```python
图表类型: 分组柱状图（Before vs After）

数据源: vbp_impact.csv + demand_daily.csv

X轴: 月份（集采前6个月 ~ 集采后6个月）
Y轴: 销量（盒）

分组:
  - 蓝色柱: 实际历史销量（集采前）
  - 红色柱: 模拟预测销量（集采后）

关键标记:
  - 在集采执行月份画一条竖虚线
  - 标注: "集采执行: 价格下降51%"

文字说明:
  "集采后预计销量上升30-50%（降价→更多患者用得起）"
```

### 4.3 医院→零售需求转移模拟

```python
概念: 集采中选药品在医院渠道销量下降（因为医院优先用中选品种），
      但非中选替代品在零售渠道销量上升。

图表类型: 双Y轴折线图

线条:
  - 左Y轴（销量）: 中选药品 hospital渠道销量 ↘
  - 右Y轴（销量）: 替代品 chain+independent渠道销量 ↗

说明文字:
  "中选药品在医院销量下降，但零售药店销量上升（患者自购）"
```

### 4.4 收入影响估算

```python
组件: 计算卡片

输入:
  - 当前月销量: XXX盒
  - 当前单价: ¥YY.YY
  - 集采后单价: ¥ZZ.ZZ（降价51%）
  - 预计销量增幅: +45%

计算:
  当前月收入 = 销量 × 单价
  集采后收入 = 销量 × 1.45 × 单价 × 0.49
  收入变化 = 集采后收入 - 当前月收入

展示:
  "当前月收入: ¥298,000"
  "集采后月收入: ¥213,145"
  "变化: -¥84,855 (-28.5%)"
  "虽然单价下降，但销量上升部分抵消了收入损失"
```

---

## 页面5: 预警通知 (Alerts)

### 5.1 缺货预警

```python
数据源: xgboost_predictions.csv + replenishment.csv

逻辑:
  当前库存 < 安全库存 → 未来5-7天内可能断货

展示: 红色警告卡片列表
  🔴 [SKU-0001 氨氯地平片] 预计3天后断货！当前库存: 25盒，建议立即补货306盒
  🔴 [SKU-0045 二甲双胍片] 预计5天后断货！当前库存: 12盒，建议立即补货150盒
```

### 5.2 库存积压预警

```python
数据源: inventory.csv + demand_daily.csv

逻辑:
  库存周转天数 > 180天 AND 近90天销量极低

展示: 黄色警告卡片列表
  🟡 [SKU-0234 XX药] 库存积压！已存放200天，近90天仅售出2盒
  建议: 促销清仓或退货处理
```

---

## 全局UI要求

### 布局
- **Sidebar**: 页面导航 + 全局筛选（日期范围、治疗领域）
- **Main Area**: 当前页面的图表和内容
- **Header**: 应用标题 + 当前数据更新时间

### 样式
- 主题色：蓝色系（#1f77b4为主色）
- 警告色：红色（#d62728）、黄色（#ff7f0e）、绿色（#2ca02c）
- 字体：中文优先使用系统默认无衬线字体
- 响应式：适配1280px以上屏幕

### Tooltip说明
所有医药专业术语都要有tooltip解释：
- **ATC**: 解剖学治疗学化学分类系统，WHO药品分类标准
- **VBP/集采**: 国家药品集中采购，批量招标降价
- **SKU**: 库存保有单位，一种药品+规格=1个SKU
- **MAPE**: 平均绝对百分比误差，衡量预测准确度的指标
- **FEFO**: 先到期先出，药品库存管理规则

---

## 数据来源速查表

| 前端展示内容 | 数据来源文件 | 关键字段 |
|-------------|-------------|----------|
| SKU选择器 | products.csv | sku_name, generic_name_cn |
| 历史销量折线 | demand_daily.csv | date, demand_total/hospital/chain/independent |
| 未来预测折线 | xgboost_predictions.csv | date, forecast_total/hospital/chain/independent |
| 预测准确度 | xgboost_metrics.csv | mape, rmse |
| 特征重要性 | xgboost_feature_importance.csv | feature, importance |
| 补货建议 | replenishment.csv | suggested_order_qty, priority, safety_stock |
| ABC/XYZ矩阵 | demand_daily.csv + products.csv | 销售额, CV |
| 近效期预警 | products.csv + inventory.csv | shelf_life_months, ending_inventory |
| 集采模拟 | vbp_impact.csv + demand_daily.csv | pre/post_vbp_price, volume_uplift_pct |

---

*本文档基于老师PLAN.md (Phase 3) 要求编写*
*技术栈: Streamlit + Plotly + Pandas*
