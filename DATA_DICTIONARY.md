# 数据字典说明书 (Data Dictionary)

> 本文档详细说明项目中每个数据文件的用途、字段含义和业务逻辑。
> 
> 生成工具: `01_generate_data.py` (Python合成数据生成器)
> 时间范围: 2020-01-01 至 2026-05-31
> SKU数量: 500个
> 渠道: 总体 / 医院 / 连锁药店 / 独立药店

---

## 文件总览

| 文件名 | 用途 | 行数 | 核心内容 |
|--------|------|------|----------|
| `products.csv` | SKU主数据 | 500 | 药品属性、ATC编码、价格等 |
| `demand_daily.csv` | 日度需求 | 1,171,500 | 每日各渠道销量 |
| `external_signals.csv` | 外部信号 | 2,343 | 流感、假期、集采等 |
| `holidays.csv` | 假期日历 | 33 | 中国法定假期 |
| `sku_profiles.csv` | SKU需求画像 | 500 | 需求分类(fast/seasonal等) |
| `vbp_impact.csv` | 集采冲击 | ~150 | 集采前后价格/量变化 |
| `inventory.csv` | 日度库存 | 1,171,500 | 每日期末库存、缺货标记 |
| `replenishment.csv` | 补货建议 | 500 | 再订货点、建议订货量 |

---

## 1. products.csv — SKU主数据

**每行含义**: 一个药品SKU的基本属性信息

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `sku_id` | SKU编码 | 字符串 | SKU-0001 | 唯一标识符 |
| `atc_code` | ATC编码(5级) | 字符串 | C08CA01 | 完整ATC五级编码 |
| `atc_level1` | ATC一级 | 字符串 | C | 解剖学大类(如C=心血管) |
| `atc_level2` | ATC二级 | 字符串 | C08 | 治疗学亚类(如C08=钙通道阻滞剂) |
| `atc_level3` | ATC三级 | 字符串 | C08C | 药理学亚类 |
| `atc_level4` | ATC四级 | 字符串 | C08CA | 化学亚类 |
| `product_name` | 产品英文名 | 字符串 | Amlodipine Tablets 5mgx28 | 英文名称+规格 |
| `sku_name` | 产品中文名 | 字符串 | 氨氯地平片 5mgx28 | 中文名称+规格 |
| `generic_name` | 通用名(英) | 字符串 | Amlodipine | WHO通用名 |
| `generic_name_cn` | 通用名(中) | 字符串 | 氨氯地平 | 中文通用名 |
| `category` | 品类 | 字符串 | C08C | 品类分类(同ATC-3) |
| `therapy_area` | 治疗领域 | 字符串 | Cardiovascular | 治疗大类(6类之一) |
| `dosage_form` | 剂型 | 字符串 | Tablets | 片剂/胶囊/颗粒等 |
| `specification` | 规格 | 字符串 | 5mgx28 | 规格+包装 |
| `is_rx` | 是否处方药 | 整数(0/1) | 1 | 1=处方药,0=OTC |
| `is_essential_medicine` | 是否基药 | 整数(0/1) | 1 | 1=国家基本药物 |
| `vbp_flag` | 是否集采 | 整数(0/1) | 1 | 1=纳入集采 |
| `vbp_batch` | 集采批次 | 整数 | 1 | 第几批国家集采 |
| `unit_price_cny` | 单价(元) | 浮点数 | 14.50 | 当前售价(集采后) |
| `shelf_life_months` | 保质期(月) | 整数 | 36 | 36个月=3年 |
| `storage_type` | 存储条件 | 字符串 | Ambient | Ambient常温/Cold Chain冷链 |
| `cold_chain` | 是否冷链 | 整数(0/1) | 0 | 1=需冷链运输 |
| `supplier` | 供应商 | 字符串 | 辉瑞制药 | 生产企业/经销商 |
| `lead_time_days` | 配送提前期 | 整数 | 7 | 下单到收货的天数 |

### ATC编码说明

ATC = Anatomical Therapeutic Chemical（解剖学治疗学化学分类系统）

| 层级 | 编码 | 含义 | 示例 |
|------|------|------|------|
| ATC-1 | C | 解剖学主类 | C = 心血管系统 |
| ATC-2 | C08 | 治疗学亚类 | C08 = 钙通道阻滞剂 |
| ATC-3 | C08C | 药理学亚类 | C08C = 选择性钙通道阻滞剂 |
| ATC-4 | C08CA | 化学亚类 | C08CA = 二氢吡啶衍生物 |
| ATC-5 | C08CA01 | 具体药品 | C08CA01 = 氨氯地平 |

### 6大治疗领域

| 治疗领域 | 英文 | 主要药品类别 |
|----------|------|-------------|
| 心血管 | Cardiovascular | 降压药、降脂药、抗心律失常 |
| 抗生素 | Antibiotics | 青霉素类、头孢类、大环内酯 |
| 糖尿病 | Diabetes | 口服降糖药、胰岛素、DPP-4抑制剂 |
| 呼吸系统 | Respiratory | 抗病毒、平喘药、祛痰药 |
| 中枢神经系统 | CNS | 抗抑郁、抗痴呆、抗癫痫 |
| 消化系统 | Gastrointestinal | 抑酸药、促胃肠动力、益生菌 |

---

## 2. demand_daily.csv — 日度需求数据

**每行含义**: 某一天某一个SKU的全国总销量（拆分为4个渠道）

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `date` | 日期 | 日期 | 2020-01-01 | 日度粒度 |
| `sku_id` | SKU编码 | 字符串 | SKU-0001 | 关联products.csv |
| `demand_total` | 总需求量 | 整数 | 150 | 全国4渠道合计销量(盒) |
| `demand_hospital` | 医院渠道 | 整数 | 60 | 公立医院销量 |
| `demand_chain` | 连锁药店 | 整数 | 52 | 连锁药店销量 |
| `demand_independent` | 独立药店 | 整数 | 38 | 单体/独立药店销量 |

### 渠道拆分逻辑

```
demand_total = demand_hospital + demand_chain + demand_independent
```

每个SKU的4渠道比例不同：
- **处方药**（is_rx=1）: 医院渠道占比高（30-45%），因为需处方
- **OTC药**: 药店渠道占比高（70-80%）
- **连锁 vs 独立**: 一般6:4或7:3

### 数据规模

- 总行数: 500 SKU × 2,343天 = **1,171,500行**
- 文件大小: ~35MB
- 时间跨度: 2020-01-01 至 2026-05-31（2,343天）

---

## 3. external_signals.csv — 外部信号

**每行含义**: 某一天的宏观外部影响因素

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `date` | 日期 | 日期 | 2020-01-01 | 日度粒度 |
| `year` | 年份 | 整数 | 2020 | — |
| `month` | 月份 | 整数 | 1 | 1-12 |
| `day` | 日期 | 整数 | 1 | 1-31 |
| `day_of_week` | 星期 | 整数 | 2 | 0=周一,6=周日 |
| `week_of_year` | 年内第几周 | 整数 | 1 | 1-53 |
| `quarter` | 季度 | 整数 | 1 | 1-4 |
| `is_weekend` | 是否周末 | 整数(0/1) | 0 | 周末销量通常略低 |
| `is_holiday` | 是否假期 | 整数(0/1) | 1 | 假期药店关门→销量降 |
| `days_to_nearest_holiday` | 距最近假期天数 | 整数 | 0 | 节前备货小高峰 |
| `season` | 季节 | 字符串 | winter | spring/summer/autumn/winter |
| `is_flu_season` | 是否流感季 | 整数(0/1) | 1 | 11月-3月=1 |
| `flu_activity_index` | 流感活动指数 | 浮点数 | 6.5 | 0-10，模拟CDC ILI% |
| `is_allergy_season` | 是否过敏季 | 整数(0/1) | 0 | 3-5月=1 |
| `baidu_flu_index` | 百度流感搜索指数 | 整数 | 5200 | 模拟百度指数 |
| `is_vbp_period` | 是否集采期 | 整数(0/1) | 0 | 集采执行后=1 |
| `days_since_vbp1` | 距首批集采天数 | 整数 | -1 | -1=未开始，>=0=已执行 |
| `days_since_vbp2` | 距二批集采天数 | 整数 | -1 | 同上 |

### 外部信号的业务含义

| 信号 | 来源 | 影响机制 |
|------|------|----------|
| **流感活动指数** | 模拟CDC ILI% | 流感季→呼吸道药品需求↑→部分慢性病药（心血管）也↑ |
| **百度流感搜索** | 模拟百度指数 | 搜索量↑→预示后续就诊量↑→药品需求↑（先行指标）|
| **假期效应** | 中国法定假期 | 假期药店关门→销量↓；节前备货→销量↑ |
| **集采标记** | VBP批次日期 | 集采执行→降价→需求量结构性跃升 |

---

## 4. holidays.csv — 假期日历

**每行含义**: 一个法定假期

| 字段名 | 中文名 | 示例值 | 说明 |
|--------|--------|--------|------|
| `date` | 日期 | 2020-01-24 | 假期当天 |
| `year` | 年份 | 2020 | — |
| `month` | 月份 | 1 | — |
| `day` | 日期 | 24 | — |
| `holiday_name` | 假期名称 | Spring Festival | 春节/清明/五一/端午/国庆 |

---

## 5. sku_profiles.csv — SKU需求画像

**每行含义**: 一个SKU的需求特征分类（决定用哪个预测模型）

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `sku_id` | SKU编码 | 字符串 | SKU-0001 | — |
| `demand_class` | 需求分类 | 字符串 | fast | fast/seasonal/long_tail/policy_shocked |
| `base_demand` | 基础日均需求 | 浮点数 | 250.0 | 模拟的日均销量基线 |
| `demand_cv` | 需求变异系数 | 浮点数 | 0.15 | CV越高=波动越大 |
| `cold_chain` | 是否冷链 | 整数(0/1) | 0 | — |
| `vbp_flag` | 是否集采 | 整数(0/1) | 1 | — |
| `vbp_shock_month` | 集采冲击月 | 字符串 | 2021-06-01 | 集采开始日期 |

### 需求分类详解

| 分类 | 占比 | 特征 | 预测模型 |
|------|------|------|----------|
| **fast** | ~8% | 销量高且稳定，CV<0.3 | ETS指数平滑 |
| **seasonal** | ~9% | 有明显季节性波动 | Prophet季节性分解 |
| **long_tail** | ~80% | 销量极低，经常0需求 | Croston间歇需求 |

### 分类依据

```
if CV < 0.3 and 销量高:   → fast
elif 有季节性模式:         → seasonal
elif 受VBP影响:            → policy_shocked
else (CV高, 销量低):       → long_tail
```

---

## 6. vbp_impact.csv — 集采冲击参数

**每行含义**: 一个纳入集采的SKU的价格和销量变化

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `sku_id` | SKU编码 | 字符串 | SKU-0001 | — |
| `pre_vbp_price` | 集采前价格 | 浮点数 | 29.80 | 元/盒 |
| `post_vbp_price` | 集采后价格 | 浮点数 | 14.50 | 降价51%后 |
| `price_drop_pct` | 降价幅度 | 浮点数 | -51.0 | 百分比 |
| `volume_uplift_pct` | 量增幅度 | 浮点数 | 45.2 | 降价后需求量上升的百分比 |
| `vbp_batch` | 集采批次 | 整数 | 1 | 第几批 |
| `vbp_date` | 集采日期 | 字符串 | 2021-06-01 | 全国执行日期 |

### 集采冲击机制

```
集采前: 价格 = 29.8元, 日均销量 = 100盒
集采后: 价格 = 14.5元(-51%), 日均销量 = 145盒(+45%)

原因: 降价 → 更多患者用得起 → 处方量增加 → 总需求量上升
```

---

## 7. inventory.csv — 日度库存

**每行含义**: 某一天某一个SKU的期末库存状态

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `date` | 日期 | 日期 | 2020-01-01 | — |
| `sku_id` | SKU编码 | 字符串 | SKU-0001 | — |
| `ending_inventory` | 期末库存 | 整数 | 5,200 | 当天结束时的库存量(盒) |
| `receipt_qty` | 到货量 | 整数 | 0 | 当天到货的数量 |
| `stockout_flag` | 缺货标记 | 整数(0/1) | 0 | 1=当天发生缺货 |

### 库存模拟逻辑

```
每日更新:
  如果 期末库存 < 再订货点:
     触发订货 → lead_time天后到货
  
  期末库存 = 上期库存 + 到货 - 销售
  
  如果 销售 > 上期库存:
     缺货标记 = 1
     期末库存 = 0
```

---

## 8. replenishment.csv — 补货建议基线

**每行含义**: 一个SKU的当前补货状态和建议

| 字段名 | 中文名 | 类型 | 示例值 | 说明 |
|--------|--------|------|--------|------|
| `sku_id` | SKU编码 | 字符串 | SKU-0001 | — |
| `current_inventory` | 当前库存 | 整数 | 3,500 | 最新期末库存 |
| `avg_monthly_demand` | 月均需求 | 整数 | 7,500 | 过去30天日均×30 |
| `safety_stock` | 安全库存 | 整数 | 3,750 | 半月需求量 |
| `reorder_point` | 再订货点 | 整数 | 1,125 | 提前期需求量×1.5 |
| `suggested_order_qty` | 建议订货量 | 整数 | 6,375 | (提前期+14天)×日均 - 当前库存 |
| `order_value_cny` | 订货金额 | 浮点数 | 92,437.50 | 建议订货量×单价 |
| `priority` | 优先级 | 字符串 | Medium | High/Medium/Low |
| `lead_time_days` | 配送提前期 | 整数 | 7 | 下单到收货天数 |

### 优先级判断

| 条件 | 优先级 | 含义 |
|------|--------|------|
| 当前库存 < 安全库存 | **High** | 紧急补货，可能即将缺货 |
| 当前库存 < 再订货点 | **Medium** | 需要补货 |
| 当前库存 ≥ 再订货点 | **Low** | 库存充足 |

---

## 附录A：数据生成原理

### 需求引擎公式

```
demand_total = base_demand × trend × seasonality × flu_effect × vbp_effect × holiday_effect × noise

其中:
  trend        = 长期增长趋势 (年化3-8%)
  seasonality  = 季节性因子 (冬季1.05, 夏季0.95)
  flu_effect   = 1 + 0.05 × flu_activity_index/10
  vbp_effect   = 集采后价格下降→需求上升 (+30~60%)
  holiday      = 假期0.3 (关门), 节前1.2 (备货)
  noise        = 对数正态噪声 (CV控制波动大小)
```

### 渠道拆分

```
demand_total = demand_hospital + demand_chain + demand_independent

hospital_ratio      = 25-45% (处方药高，OTC低)
chain_ratio         = 50-70% × (1 - hospital_ratio)
independent_ratio   = 剩余部分
```

---

## 附录B：预测模型与数据的关系

| 模型 | 使用的数据文件 | 关键特征 | 适用SKU |
|------|--------------|----------|---------|
| **ETS** | demand_daily.csv | demand_total时间序列 | fast (39个) |
| **Prophet** | demand_daily + external_signals + holidays | demand_total + flu + holiday | seasonal (46个) |
| **Croston/SBA** | demand_daily.csv | demand_total(间歇性) | long_tail (399个) |
| **XGBoost** | demand_daily + external_signals + products + sku_profiles | 全特征(滞后+外部+ATC) | policy_shocked (16个) |

---

*本文档生成时间: 2026-06-04*
*数据生成种子: SEED=42（确保可复现）*
