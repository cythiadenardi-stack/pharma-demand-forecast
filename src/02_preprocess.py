"""
数据预处理与特征工程 — 02_preprocess.py
====================================================
对原始数据进行合并和特征工程，生成模型训练用的数据集。

由于数据量大（117万行），本脚本采用流式处理：
1. 逐SKU处理，避免内存溢出
2. 按demand_class分别保存，每个模型只加载自己需要的数据
3. 滞后特征按需生成

输入: data/*.csv
输出: data/processed/*_train.csv, *_val.csv, *_test.csv

运行: python 02_preprocess.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 60)
print("数据预处理与特征工程")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

# ============================================================
# 第一步：读取原始数据
# ============================================================

print("\n[1/4] 读取原始数据...")

demand = pd.read_csv("data/demand_daily.csv", parse_dates=['date'])
external = pd.read_csv("data/external_signals.csv", parse_dates=['date'])
products = pd.read_csv("data/products.csv")
sku_profiles = pd.read_csv("data/sku_profiles.csv")

print(f"    demand:        {demand.shape}")
print(f"    external:      {external.shape}")
print(f"    products:      {products.shape}")
print(f"    sku_profiles:  {sku_profiles.shape}")

# ============================================================
# 第二步：构建SKU→demand_class映射
# ============================================================

sku_to_class = dict(zip(sku_profiles['sku_id'], sku_profiles['demand_class']))
class_counts = sku_profiles['demand_class'].value_counts()
print(f"\n[2/4] SKU分类分布:")
for cls, cnt in class_counts.items():
    print(f"    {cls:20s}: {cnt:3d} SKU ({cnt/500*100:.1f}%)")

# ============================================================
# 第三步：逐SKU处理并保存
# ============================================================

print(f"\n[3/4] 逐SKU处理（共{len(sku_profiles)}个SKU）...")

# 日期划分
train_end = pd.Timestamp("2024-06-30")
val_end = pd.Timestamp("2025-02-28")

# 为每个demand_class准备一个写入器
class_writers = {}
for dclass in sku_profiles['demand_class'].unique():
    for split in ['train', 'val', 'test']:
        filepath = f"data/processed/{dclass}_{split}.csv"
        class_writers[(dclass, split)] = filepath
        # 创建空文件（写入表头）
        open(filepath, 'w').close()

# 逐SKU处理（流式，内存友好）
processed = 0
header_written = {}

for sku_id in sku_profiles['sku_id']:
    # 1. 取出该SKU的需求数据
    sku_demand = demand[demand['sku_id'] == sku_id].copy()
    if len(sku_demand) == 0:
        continue
    
    # 2. 合并外部信号
    sku_data = sku_demand.merge(external, on='date', how='left')
    
    # 3. 合并产品属性
    sku_prod = products[products['sku_id'] == sku_id]
    if len(sku_prod) > 0:
        for col in ['atc_level1', 'atc_level2', 'atc_level3', 'atc_level4',
                     'therapy_area', 'vbp_flag', 'unit_price_cny', 'lead_time_days']:
            sku_data[col] = sku_prod[col].values[0]
    
    # 4. 合并profile
    sku_prof = sku_profiles[sku_profiles['sku_id'] == sku_id]
    dclass = sku_prof['demand_class'].values[0]
    sku_data['demand_class'] = dclass
    sku_data['base_demand'] = sku_prof['base_demand'].values[0]
    sku_data['demand_cv'] = sku_prof['demand_cv'].values[0]
    
    # 5. 特征工程（滞后特征）
    sku_data = sku_data.sort_values('date')
    sku_data['sales_qty_lag_7'] = sku_data['demand_total'].shift(7)
    sku_data['sales_qty_lag_14'] = sku_data['demand_total'].shift(14)
    sku_data['sales_qty_lag_30'] = sku_data['demand_total'].shift(30)
    sku_data['sales_rolling_mean_7'] = sku_data['demand_total'].shift(1).rolling(7, min_periods=1).mean()
    sku_data['sales_rolling_mean_30'] = sku_data['demand_total'].shift(1).rolling(30, min_periods=1).mean()
    
    # 6. 时间编码
    sku_data['month_sin'] = np.sin(2 * np.pi * sku_data['month'] / 12)
    sku_data['month_cos'] = np.cos(2 * np.pi * sku_data['month'] / 12)
    
    # 7. VBP特征
    sku_data['is_post_vbp1'] = (sku_data['days_since_vbp1'] >= 0).astype(int)
    
    # 8. 训练/验证/测试划分
    sku_data['split'] = 'train'
    sku_data.loc[sku_data['date'] > train_end, 'split'] = 'val'
    sku_data.loc[sku_data['date'] > val_end, 'split'] = 'test'
    
    # 9. 保存
    for split in ['train', 'val', 'test']:
        split_df = sku_data[sku_data['split'] == split]
        if len(split_df) == 0:
            continue
        
        filepath = class_writers[(dclass, split)]
        
        # 第一次写入时写表头
        write_header = not header_written.get((dclass, split), False)
        split_df.to_csv(filepath, mode='a', header=write_header, index=False)
        header_written[(dclass, split)] = True
    
    processed += 1
    if processed % 100 == 0:
        print(f"    ... 已处理 {processed}/500 SKU")

print(f"    完成! 共处理 {processed} 个SKU")

# ============================================================
# 第四步：统计输出
# ============================================================

print("\n[4/4] 输出统计:")
for dclass in sorted(sku_profiles['demand_class'].unique()):
    for split in ['train', 'val', 'test']:
        filepath = class_writers[(dclass, split)]
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            # 只读第一行来计数
            with open(filepath) as f:
                n_lines = sum(1 for _ in f) - 1  # 减去表头
            size = os.path.getsize(filepath)
            size_str = f"{size/1024:.0f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
            print(f"    {dclass}_{split}.csv: {n_lines:>7,}行  {size_str}")

print("\n" + "=" * 60)
print("预处理完成！")
print("=" * 60)
