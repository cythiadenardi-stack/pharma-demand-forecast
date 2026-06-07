#!/usr/bin/env python
"""
Croston间歇需求预测模型 — 03_model_croston.py
================================================
针对 long_tail 类型SKU的间歇性需求预测。

核心设计：
- 覆盖全部399个long_tail SKU（不是100个）
- 对4个渠道分别预测
- Croston方法 + SBA改进版

Croston方法原理：
- 间歇需求：大部分时间0，偶尔有正值
- 传统方法会把0值平均掉，导致预测偏低
- Croston分离"需求间隔"和"需求大小"分别预测
- 最终预测 = 预测需求大小 / 预测需求间隔

输入: data/demand_daily.csv, data/sku_profiles.csv
输出: results/croston_predictions.csv, results/croston_metrics.csv

运行: python src/03_model_croston.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings('ignore')

# ============================================================
# 第一部分：全局配置
# ============================================================

# 时间划分
TRAIN_END = pd.Timestamp("2024-12-31")
FORECAST_START = pd.Timestamp("2026-06-01")
FORECAST_END = pd.Timestamp("2026-08-30")

# 4个渠道
CHANNELS = ['demand_total', 'demand_hospital', 'demand_chain', 'demand_independent']
CHANNEL_NAMES = {'demand_total': 'total', 'demand_hospital': 'hospital',
                 'demand_chain': 'chain', 'demand_independent': 'independent'}

# Croston平滑系数
ALPHA = 0.1

print("=" * 60)
print("Croston间歇需求预测模型（全部399 long_tail SKU × 4渠道）")
print("=" * 60)

# ============================================================
# 第二部分：Croston方法核心算法
# ============================================================

def croston_forecast(demand_series, alpha=0.1):
    """
    Croston方法核心算法
    
    参数:
        demand_series: 历史需求序列（numpy数组或列表）
        alpha: 平滑系数（默认0.1）
    
    返回:
        forecast: 下一个周期的预测值
        p: 最终的需求间隔估计
        q: 最终的需求大小估计
    """
    n = len(demand_series)
    if n == 0:
        return 0, 1, 0
    
    # 初始化
    p = 1.0   # 需求间隔（两个非零需求之间的平均天数）
    q = demand_series[0] if demand_series[0] > 0 else 0  # 需求大小
    
    last_demand_idx = 0  # 上次有需求的索引
    first_demand = True  # 标记是否是第一个非零需求
    
    for i in range(1, n):
        if demand_series[i] > 0:
            # 计算距离上次需求的时间间隔
            if first_demand:
                interval = 1  # 第一个间隔设为1
                first_demand = False
            else:
                interval = i - last_demand_idx
            
            # 用指数平滑更新需求间隔
            p = alpha * interval + (1 - alpha) * p
            # 用指数平滑更新需求大小
            q = alpha * demand_series[i] + (1 - alpha) * q
            
            last_demand_idx = i
    
    # 最终预测 = 预测需求大小 / 预测需求间隔
    forecast = q / p if p > 0 else 0
    return forecast, p, q


def sba_forecast(demand_series, alpha=0.1):
    """
    SBA改进版 (Syntetos-Boylan Approximation)
    
    原理：Croston方法有正偏差（总是高估），SBA通过乘以一个修正因子来纠正
    SBA = Croston * (1 - alpha/2)
    """
    croston_val, p, q = croston_forecast(demand_series, alpha)
    sba_val = croston_val * (1 - alpha / 2)
    return sba_val, croston_val, p, q


# ============================================================
# 第三部分：数据加载
# ============================================================

print("\n[1/4] 加载数据...")

demand = pd.read_csv("data/demand_daily.csv", parse_dates=['date'])
sku_profiles = pd.read_csv("data/sku_profiles.csv")

# 筛选long_tail SKU
long_tail_skus = sku_profiles[sku_profiles['demand_class'] == 'long_tail']['sku_id'].tolist()
print(f"    long_tail SKU数量: {len(long_tail_skus)}")

# ============================================================
# 第四部分：主流程 — 全部399个SKU × 4渠道
# ============================================================

print("\n[2/4] 开始预测（全部399 SKU × 4渠道）...")

os.makedirs("results", exist_ok=True)

predictions = []
metrics = []

start_time = time.time()
processed = 0
total = len(long_tail_skus) * len(CHANNELS)

for sku_id in long_tail_skus:
    for channel in CHANNELS:
        processed += 1
        
        if processed % 200 == 0:
            elapsed = time.time() - start_time
            print(f"    ... {processed}/{total} ({elapsed:.0f}秒)")
        
        # 4.1 取出该SKU该渠道的历史需求
        sku_data = demand[(demand['sku_id'] == sku_id) & (demand['date'] <= TRAIN_END)]
        demand_series = sku_data[channel].values
        
        if len(demand_series) == 0:
            continue
        
        # 4.2 用Croston+SBA预测
        sba_val, croston_val, p, q = sba_forecast(demand_series, ALPHA)
        
        # 4.3 评估（用训练集最后30天）
        test_data = demand[(demand['sku_id'] == sku_id) & 
                           (demand['date'] > TRAIN_END) & 
                           (demand['date'] < FORECAST_START)]
        
        if len(test_data) > 0:
            y_true = test_data[channel].values
            y_pred_croston = np.full_like(y_true, croston_val, dtype=float)
            y_pred_sba = np.full_like(y_true, sba_val, dtype=float)
            
            # RMSE
            rmse_croston = np.sqrt(np.mean((y_true - y_pred_croston) ** 2))
            rmse_sba = np.sqrt(np.mean((y_true - y_pred_sba) ** 2))
            
            # MAE
            mae_croston = np.mean(np.abs(y_true - y_pred_croston))
            mae_sba = np.mean(np.abs(y_true - y_pred_sba))
            
            # MAPE（只在非零点计算）
            nonzero_mask = y_true > 0
            if nonzero_mask.sum() > 0:
                mape_croston = np.mean(np.abs((y_true[nonzero_mask] - y_pred_croston[nonzero_mask]) / y_true[nonzero_mask])) * 100
                mape_sba = np.mean(np.abs((y_true[nonzero_mask] - y_pred_sba[nonzero_mask]) / y_true[nonzero_mask])) * 100
            else:
                mape_croston = np.nan
                mape_sba = np.nan
            
            metrics.append({
                'sku_id': sku_id,
                'channel': CHANNEL_NAMES[channel],
                'nonzero_ratio': round(nonzero_mask.mean(), 3),
                'croston_forecast': round(croston_val, 4),
                'sba_forecast': round(sba_val, 4),
                'rmse_croston': round(rmse_croston, 4),
                'rmse_sba': round(rmse_sba, 4),
                'mae_croston': round(mae_croston, 4),
                'mae_sba': round(mae_sba, 4),
                'mape_croston': round(mape_croston, 2) if not np.isnan(mape_croston) else None,
                'mape_sba': round(mape_sba, 2) if not np.isnan(mape_sba) else None,
            })
        
        # 4.4 保存未来90天预测（Croston是点预测，每天相同）
        forecast_days = (FORECAST_END - FORECAST_START).days + 1
        for day_offset in range(forecast_days):
            pred_date = FORECAST_START + timedelta(days=day_offset)
            predictions.append({
                'sku_id': sku_id,
                'date': pred_date.strftime('%Y-%m-%d'),
                'channel': CHANNEL_NAMES[channel],
                'croston_forecast': round(croston_val, 4),
                'sba_forecast': round(sba_val, 4),
            })

elapsed = time.time() - start_time
print(f"\n    完成: {processed}个SKU+渠道组合, 耗时{elapsed:.0f}秒")

# ============================================================
# 第五部分：保存结果
# ============================================================

print("\n[3/4] 保存结果...")

# 5.1 预测结果 — 转成宽格式
pred_df = pd.DataFrame(predictions)
if len(pred_df) > 0:
    # 每个渠道两列预测值
    pred_pivot = pred_df.pivot_table(
        index=['sku_id', 'date'],
        columns='channel',
        values=['croston_forecast', 'sba_forecast'],
        aggfunc='first'
    ).reset_index()
    pred_pivot.columns.name = None
    # 展平列名
    pred_pivot.columns = [' '.join(col).strip() if col[1] not in ['nan', ''] else col[0] 
                          for col in pred_pivot.columns.values]
    pred_pivot.to_csv("results/croston_predictions.csv", index=False)
    print(f"    [OK] croston_predictions.csv: {len(pred_pivot)}行")
else:
    print("    ✗ 无预测结果")

# 5.2 评估指标
if len(metrics) > 0:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("results/croston_metrics.csv", index=False)
    print(f"    [OK] croston_metrics.csv: {len(metrics_df)}行")
    
    # SBA vs Croston改进
    mae_cro = metrics_df['mae_croston'].mean()
    mae_sba = metrics_df['mae_sba'].mean()
    improvement = (mae_cro - mae_sba) / mae_cro * 100 if mae_cro > 0 else 0
    print(f"\n    SBA vs Croston MAE改进: {improvement:.1f}%")
    print(f"    平均MAE: Croston={mae_cro:.2f}, SBA={mae_sba:.2f}")
    
    # 按渠道汇总
    for ch in ['total', 'hospital', 'chain', 'independent']:
        ch_df = metrics_df[metrics_df['channel'] == ch]
        if len(ch_df) > 0:
            print(f"      {ch:15s}: SKU数={len(ch_df)}, 平均MAE={ch_df['mae_sba'].mean():.2f}")

print("\n" + "=" * 60)
print("Croston模型完成！")
print(f"  覆盖SKU: {len(long_tail_skus)}个 (全部long_tail)")
print(f"  覆盖渠道: 4个")
print(f"  预测天数: {(FORECAST_END - FORECAST_START).days + 1}天")
print(f"  说明: Croston是点预测(30天相同), SBA是改进版")
print("=" * 60)
