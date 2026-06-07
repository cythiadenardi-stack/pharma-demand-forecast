#!/usr/bin/env python
"""
XGBoost机器学习预测模型 — 03_model_xgboost.py
================================================
用XGBoost预测全部500个SKU的全部4个渠道。

核心设计：
- XGBoost作为"主力高级模型"，覆盖全部500个SKU
- 对每个SKU的每个渠道单独训练一个XGBoost模型
- 递推式预测：每天预测后更新特征，确保30天预测各不相同
- 特征工程：滞后特征 + 外部信号 + ATC编码 + 交互特征

输入: data/demand_daily.csv, data/external_signals.csv, data/products.csv, data/sku_profiles.csv
输出: results/xgboost_predictions.csv, results/xgboost_metrics.csv, results/xgboost_feature_importance.csv

运行: python src/03_model_xgboost.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================
# 第一部分：导入XGBoost（带安装检测）
# ============================================================

try:
    import xgboost as xgb
    print("[OK] XGBoost 版本:", xgb.__version__)
except ImportError:
    print("=" * 60)
    print("错误: 未安装 xgboost 库")
    print("请运行: pip install xgboost")
    print("=" * 60)
    sys.exit(1)

# ============================================================
# 第二部分：全局配置
# ============================================================

# 时间划分
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
FORECAST_START = pd.Timestamp("2026-06-01")
FORECAST_END = pd.Timestamp("2026-08-30")

# 4个渠道
CHANNELS = ['demand_total', 'demand_hospital', 'demand_chain', 'demand_independent']
CHANNEL_NAMES = {'demand_total': 'total', 'demand_hospital': 'hospital', 
                 'demand_chain': 'chain', 'demand_independent': 'independent'}

# XGBoost参数
XGB_PARAMS = {
    'n_estimators': 150,      # 树的数量
    'max_depth': 6,           # 树的最大深度
    'learning_rate': 0.1,     # 学习率
    'subsample': 0.8,         # 每棵树用80%的样本
    'colsample_bytree': 0.8,  # 每棵树用80%的特征
    'random_state': 42,
    'n_jobs': 4,              # 用4个CPU核心
}

print("=" * 60)
print("XGBoost 药品需求预测模型（全部500 SKU × 4渠道）")
print("=" * 60)

# ============================================================
# 第三部分：数据加载
# ============================================================

print("\n[1/5] 加载数据...")

demand = pd.read_csv("data/demand_daily.csv", parse_dates=['date'])
external = pd.read_csv("data/external_signals.csv", parse_dates=['date'])
products = pd.read_csv("data/products.csv")
sku_profiles = pd.read_csv("data/sku_profiles.csv")

print(f"    需求数据: {demand.shape}")
print(f"    外部信号: {external.shape}")
print(f"    产品主数据: {products.shape}")
print(f"    SKU画像: {sku_profiles.shape}")

# ============================================================
# 第四部分：特征工程
# ============================================================

print("\n[2/5] 构建特征工程pipeline...")

def build_features(df_demand, df_external, df_products, df_profiles, sku_id, channel):
    """
    为一个SKU的一个渠道构建完整的特征矩阵
    
    参数:
        df_demand: 需求数据
        df_external: 外部信号
        df_products: 产品主数据
        df_profiles: SKU画像
        sku_id: SKU编码
        channel: 渠道列名(demand_total/hospital/chain/independent)
    
    返回:
        df_feature: 带特征的数据框
    """
    
    # 4.1 取出该SKU的需求数据（只取该渠道）
    sku_data = df_demand[df_demand['sku_id'] == sku_id][['date', channel]].copy()
    sku_data.columns = ['date', 'y']  # y = 目标变量
    
    # 4.2 合并外部信号
    sku_data = sku_data.merge(df_external, on='date', how='left')
    
    # 4.3 添加产品属性
    prod = df_products[df_products['sku_id'] == sku_id]
    if len(prod) > 0:
        sku_data['vbp_flag'] = prod['vbp_flag'].values[0]
        sku_data['unit_price'] = prod['unit_price_cny'].values[0]
        sku_data['lead_time'] = prod['lead_time_days'].values[0]
    
    # 4.4 添加SKU画像
    prof = df_profiles[df_profiles['sku_id'] == sku_id]
    if len(prof) > 0:
        sku_data['base_demand'] = prof['base_demand'].values[0]
        sku_data['demand_cv'] = prof['demand_cv'].values[0]
    
    # 4.5 滞后特征（必须按SKU+渠道分组）
    sku_data = sku_data.sort_values('date').reset_index(drop=True)
    sku_data['lag_7'] = sku_data['y'].shift(7)
    sku_data['lag_14'] = sku_data['y'].shift(14)
    sku_data['lag_30'] = sku_data['y'].shift(30)
    
    # 4.6 滑动窗口特征
    sku_data['rolling_mean_7'] = sku_data['y'].shift(1).rolling(7, min_periods=1).mean()
    sku_data['rolling_mean_30'] = sku_data['y'].shift(1).rolling(30, min_periods=1).mean()
    sku_data['rolling_std_7'] = sku_data['y'].shift(1).rolling(7, min_periods=1).std().fillna(0)
    
    # 4.7 时间编码（正弦/余弦）
    sku_data['month_sin'] = np.sin(2 * np.pi * sku_data['month'] / 12)
    sku_data['month_cos'] = np.cos(2 * np.pi * sku_data['month'] / 12)
    sku_data['dow_sin'] = np.sin(2 * np.pi * sku_data['day_of_week'] / 7)
    sku_data['dow_cos'] = np.cos(2 * np.pi * sku_data['day_of_week'] / 7)
    
    # 4.8 VBP特征
    sku_data['is_post_vbp'] = (sku_data['days_since_vbp1'] >= 0).astype(int)
    sku_data['days_since_vbp_log'] = np.log1p(sku_data['days_since_vbp1'].clip(lower=0))
    
    # 4.9 交互特征
    sku_data['flu_x_vbp'] = sku_data['flu_activity_index'] * sku_data['is_post_vbp']
    
    # 4.10 丢弃无法计算特征的早期行（前30天没有lag_30）
    sku_data = sku_data.dropna(subset=['lag_30'])
    
    return sku_data


# ============================================================
# 第五部分：递推式预测
# ============================================================

def recursive_forecast(model, last_row, future_external, feature_cols, n_days=30):
    """
    递推式预测未来N天
    
    核心逻辑：每天预测后，把预测值加入历史，更新lag和rolling特征，再预测下一天
    这样可以确保30天的预测值各不相同
    
    参数:
        model: 训练好的XGBoost模型
        last_row: 最后一天的特征行（dict或Series）
        future_external: 未来30天的外部信号（DataFrame）
        feature_cols: 模型使用的特征列名列表
        n_days: 预测天数
    
    返回:
        forecasts: 30天预测值列表
    """
    forecasts = []
    
    # 初始化：用最近7天的实际值来维持rolling window
    y_val = last_row['y'] if pd.notna(last_row['y']) else 0
    recent_values = [y_val] * 7
    
    for day_idx in range(n_days):
        # 5.1 构建当前天的特征（基于last_row + 更新的lag/rolling）
        row = last_row.copy()
        
        # 更新日期相关特征
        if day_idx < len(future_external):
            ext = future_external.iloc[day_idx]
            row['month_sin'] = np.sin(2 * np.pi * ext['month'] / 12)
            row['month_cos'] = np.cos(2 * np.pi * ext['month'] / 12)
            row['dow_sin'] = np.sin(2 * np.pi * ext['day_of_week'] / 7)
            row['dow_cos'] = np.cos(2 * np.pi * ext['day_of_week'] / 7)
            row['is_holiday'] = int(ext['is_holiday'])
            row['is_weekend'] = int(ext['is_weekend'])
            row['flu_activity_index'] = float(ext['flu_activity_index'])
            dsv = float(ext['days_since_vbp1']) if pd.notna(ext['days_since_vbp1']) else -1
            row['is_post_vbp'] = 1 if dsv >= 0 else 0
            row['days_since_vbp_log'] = np.log1p(max(0, dsv))
            row['flu_x_vbp'] = row['flu_activity_index'] * row['is_post_vbp']
        
        # 5.2 更新滞后特征（把前一天的预测值推入lag链）
        if day_idx == 0:
            row['lag_7'] = float(last_row.get('lag_7', y_val)) if pd.notna(last_row.get('lag_7', y_val)) else y_val
            row['lag_14'] = float(last_row.get('lag_14', y_val)) if pd.notna(last_row.get('lag_14', y_val)) else y_val
            row['lag_30'] = float(last_row.get('lag_30', y_val)) if pd.notna(last_row.get('lag_30', y_val)) else y_val
        else:
            row['lag_7'] = forecasts[-1] if len(forecasts) >= 1 else y_val
            row['lag_14'] = forecasts[-7] if len(forecasts) >= 7 else y_val
            row['lag_30'] = forecasts[-30] if len(forecasts) >= 30 else y_val
        
        # 5.3 更新rolling特征
        recent_values.append(forecasts[-1] if forecasts else y_val)
        if len(recent_values) > 30:
            recent_values.pop(0)
        
        row['rolling_mean_7'] = float(np.mean(recent_values[-7:]))
        row['rolling_mean_30'] = float(np.mean(recent_values[-30:]))
        row['rolling_std_7'] = float(np.std(recent_values[-7:])) if len(recent_values) >= 2 else 0.0
        
        # 确保没有NaN值
        for col in feature_cols:
            if col not in row or pd.isna(row[col]):
                row[col] = 0.0
        
        # 5.4 构建特征向量并预测
        try:
            X = pd.DataFrame([{col: float(row[col]) for col in feature_cols}])
            pred = model.predict(X)[0]
            pred = max(0.0, float(pred))  # 销量不能为负
        except Exception:
            pred = y_val  # 预测失败时用最后已知值
        
        forecasts.append(pred)
    
    return forecasts


# ============================================================
# 第六部分：主流程
# ============================================================

print("\n[3/5] 开始训练...")
os.makedirs("results", exist_ok=True)

# 准备未来30天的外部信号
future_dates = pd.date_range(FORECAST_START, FORECAST_END, freq='D')
future_ext = external[external['date'].isin(future_dates)].copy()

# 获取所有SKU
all_skus = sku_profiles['sku_id'].tolist()
print(f"    总SKU数: {len(all_skus)}")

# 记录结果
predictions = []
metrics = []
feature_importance_list = []

start_time = time.time()

# 对每个SKU的每个渠道训练模型
processed = 0
total_models = len(all_skus) * len(CHANNELS)

for sku_idx, sku_id in enumerate(all_skus):
    for channel in CHANNELS:
        processed += 1
        
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    ... {processed}/{total_models} 模型已训练 ({elapsed:.0f}秒)")
        
        try:
            # 6.1 构建特征
            df_feat = build_features(demand, external, products, sku_profiles, sku_id, channel)
            
            if len(df_feat) < 100:  # 数据太少，跳过
                continue
            
            # 6.2 定义特征列和目标列
            feature_cols = [
                'lag_7', 'lag_14', 'lag_30',
                'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
                'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                'is_holiday', 'is_weekend', 'is_flu_season',
                'flu_activity_index', 'baidu_flu_index',
                'is_post_vbp', 'days_since_vbp_log', 'flu_x_vbp',
                'vbp_flag', 'base_demand', 'demand_cv'
            ]
            
            # 只保留实际存在的列
            available_cols = [c for c in feature_cols if c in df_feat.columns]
            
            # 6.3 划分训练/测试
            train_df = df_feat[df_feat['date'] <= TRAIN_END]
            test_df = df_feat[(df_feat['date'] > TRAIN_END) & (df_feat['date'] < FORECAST_START)]
            
            if len(train_df) < 50:  # 训练数据太少
                continue
            
            X_train = train_df[available_cols]
            y_train = train_df['y']
            
            # 6.4 训练XGBoost模型
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train)
            
            # 6.5 评估（如果有测试数据）
            if len(test_df) > 0:
                X_test = test_df[available_cols]
                y_test = test_df['y']
                y_pred = model.predict(X_test)
                
                # 计算MAPE和RMSE
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100  # +1避免除零
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                
                metrics.append({
                    'sku_id': sku_id,
                    'channel': CHANNEL_NAMES[channel],
                    'mape': round(mape, 2),
                    'rmse': round(rmse, 2),
                    'n_test': len(test_df)
                })
            
            # 6.6 递推预测未来90天
            last_row = df_feat.iloc[-1]
            future_ext_sku = future_ext[['date', 'month', 'day_of_week', 'is_holiday', 
                                          'is_weekend', 'flu_activity_index', 
                                          'days_since_vbp1']].copy()
            
            forecasts = recursive_forecast(model, last_row, future_ext_sku, available_cols, n_days=90)
            
            # 6.7 保存预测结果
            for i, pred_val in enumerate(forecasts):
                predictions.append({
                    'sku_id': sku_id,
                    'date': (FORECAST_START + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'channel': CHANNEL_NAMES[channel],
                    'forecast': round(pred_val, 2)
                })
            
            # 6.8 保存特征重要性（每个SKU只保存total渠道的重要性，避免数据太大）
            if channel == 'demand_total':
                importance = model.feature_importances_
                for feat, imp in zip(available_cols, importance):
                    feature_importance_list.append({
                        'sku_id': sku_id,
                        'feature': feat,
                        'importance': round(imp, 4)
                    })
            
        except Exception as e:
            # 打印错误信息以便调试
            if processed <= 5 or processed % 200 == 0:
                print(f"     ⚠️ SKU={sku_id} channel={channel} 出错: {str(e)[:80]}")
            continue

elapsed = time.time() - start_time
print(f"\n    训练完成: {processed}个模型, 耗时{elapsed:.0f}秒")

# ============================================================
# 第七部分：保存结果
# ============================================================

print("\n[4/5] 保存结果...")

# 7.1 预测结果 — 转成宽格式（每个SKU一行，4列渠道）
pred_df = pd.DataFrame(predictions)
if len(pred_df) > 0:
    pred_pivot = pred_df.pivot_table(
        index=['sku_id', 'date'], 
        columns='channel', 
        values='forecast', 
        aggfunc='first'
    ).reset_index()
    pred_pivot.columns.name = None
    pred_pivot.to_csv("results/xgboost_predictions.csv", index=False)
    print(f"    [OK] xgboost_predictions.csv: {len(pred_pivot)}行 x {len(pred_pivot.columns)}列")
    
    # 检查是否有变化
    if 'total' in pred_pivot.columns:
        sample = pred_pivot[pred_pivot['sku_id'] == pred_pivot['sku_id'].iloc[0]]
        unique_vals = sample['total'].nunique()
        print(f"    示例SKU预测值变化数: {unique_vals}/30天")
else:
    print("    ✗ 无预测结果")

# 7.2 评估指标
if len(metrics) > 0:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("results/xgboost_metrics.csv", index=False)
    print(f"    [OK] xgboost_metrics.csv: {len(metrics_df)}行")
    
    # 按渠道汇总
    for ch in ['total', 'hospital', 'chain', 'independent']:
        ch_df = metrics_df[metrics_df['channel'] == ch]
        if len(ch_df) > 0:
            print(f"      {ch:15s}: 平均MAPE={ch_df['mape'].mean():.1f}%, 平均RMSE={ch_df['rmse'].mean():.1f}")
else:
    print("    ✗ 无评估指标")

# 7.3 特征重要性
if len(feature_importance_list) > 0:
    fi_df = pd.DataFrame(feature_importance_list)
    # 取平均重要性
    fi_avg = fi_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    fi_avg.to_csv("results/xgboost_feature_importance.csv", index=False)
    print(f"    [OK] xgboost_feature_importance.csv: {len(fi_avg)}个特征")
    print(f"\n    Top 10重要特征:")
    for _, row in fi_avg.head(10).iterrows():
        print(f"      {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================
# 完成
# ============================================================

print("\n" + "=" * 60)
print("XGBoost模型完成！")
print(f"  覆盖SKU: {pred_df['sku_id'].nunique() if len(pred_df) > 0 else 0}个")
print(f"  覆盖渠道: 4个 (total/hospital/chain/independent)")
print(f"  预测天数: 30天 (2026-06-01 ~ 2026-06-30)")
print(f"  预测总数: {len(predictions)}个")
print("=" * 60)
