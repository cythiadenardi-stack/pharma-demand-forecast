# 药品需求预测与智能补货系统

> SDC MSc Innovation Management — Digital Innovation Final Assignment
> 
> 团队: Early Bird | 模块: Module 4 Case 2

## 项目概述

本项目为中国大型药品B2B电商平台构建了一套**药品需求预测与智能补货系统**原型，涵盖：

- **500个SKU** 的日度需求数据（2020-2026年）
- **4种预测模型** 按需求模式自动路由（ETS / Prophet / Croston / XGBoost）
- **4渠道拆分**（医院 / 连锁药店 / 独立药店 / 总体）
- **智能补货建议**（安全库存、再订货点、建议订货量、最佳补货时机）

## 在线演示

👉 [点击访问 Streamlit 应用](https://your-app-url.streamlit.app)

## 功能模块

1. **需求预测看板** — SKU级历史销量与90天预测趋势、渠道拆分、特征重要性分析
2. **智能补货建议** — 按渠道拆分的补货建议、优先级排序、最佳补货时机
3. **库存健康诊断** — ABC/XYZ矩阵、畅销品/滞销品识别、库存重新分配建议、近效期预警
4. **政策影响模拟** — VBP集采场景模拟、收入影响估算
5. **预警通知** — 缺货风险预警、库存积压预警

## 技术栈

- Python 3.10+
- Streamlit — 前端交互界面
- Plotly — 数据可视化
- Pandas / NumPy — 数据处理
- XGBoost / Statsmodels / Prophet — 预测模型

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 项目结构

```
.
├── app.py                  # Streamlit 前端应用
├── requirements.txt        # Python 依赖
├── data/                   # 数据文件
│   ├── products.csv
│   ├── demand_daily.csv
│   ├── inventory.csv
│   └── ...
├── results/                # 预测结果
│   ├── xgboost_predictions.csv
│   └── ...
└── src/                    # 数据生成与模型脚本
    ├── 01_generate_data.py
    └── ...
```

## 课程信息

- **课程**: SDC MSc Innovation Management — Digital Innovation
- **Case**: Case 2 — Pharmaceutical Demand Forecasting & Intelligent Replenishment
- **交付日期**: 2026-06-12
