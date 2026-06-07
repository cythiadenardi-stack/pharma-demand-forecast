#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Prophet 季节性预测模型 - 03_model_prophet.py
 ==============================================
 针对 seasonal 类型 SKU 的季节性分解预测模型，支持4个渠道分别预测。

 适用 SKU 特征: 有规律的季节波动（demand_class = 'seasonal'）
 模型: Facebook Prophet（首选）/ statsmodels SARIMAX（备选）

 关键特性:
     - 分别对 4 个渠道（总销量、医院、连锁药店、独立药店）进行训练和预测
     - 使用外部回归量（流感指数、节假日、周末）提升预测精度
     - 自动回退机制：Prophet 不可用时自动切换到 SARIMAX

 运行: python 03_model_prophet.py
 输出:
     - results/prophet_predictions.csv (预测结果，含4列渠道预测值)
     - results/prophet_metrics.csv (评估指标，按渠道分别评估)
     - results/prophet_decomposition_sku_{sku_id}_{channel}.png (趋势/季节性分解图)

 作者: 药品需求预测项目组
 日期: 2024
"""

# ============================================================
# 第一部分: 导入必要的库
# ============================================================

import os  # 操作系统接口模块，用于文件路径操作和目录管理
import sys  # 系统相关模块，用于程序退出和系统参数访问
import warnings  # 警告控制模块，用于过滤不必要的警告信息保持输出整洁
from datetime import datetime, timedelta  # 日期时间处理模块，用于时间范围计算

import numpy as np  # NumPy: 数值计算库，提供高效的数组操作和数学函数
import pandas as pd  # Pandas: 数据处理库，提供 DataFrame 等结构化数据类型

# --- Prophet 库导入处理 ---
# Prophet 是 Facebook 开发的时间序列预测工具，能自动检测趋势和季节性模式
# 如果 Prophet 未安装，则自动回退到 statsmodels 的 SARIMAX 模型作为备选方案
PROPHET_AVAILABLE = False  # 标记 Prophet 是否可用的全局标志变量，初始设为 False

try:
    # 尝试从 prophet 包导入 Prophet 类
    # Prophet 0.7+ 版本的导入方式: from prophet import Prophet
    from prophet import Prophet
    PROPHET_AVAILABLE = True  # 标记 Prophet 可用，后续将使用 Prophet 进行预测
    print("[信息] 成功加载 Prophet 库，将使用 Prophet 进行预测。")
except ImportError:
    # 当 prophet 包未安装时，捕获 ImportError 异常并输出警告信息
    print("[警告] Prophet 库未安装或导入失败。")
    print("[警告] 将使用 statsmodels 的 SARIMAX 作为备选模型。")
    print("[提示] 如需使用 Prophet，请运行: pip install prophet")

# --- SARIMAX 备选模型导入 ---
# 如果 Prophet 不可用，则导入 SARIMAX 作为备选预测模型
if not PROPHET_AVAILABLE:
    # SARIMAX: Seasonal AutoRegressive Integrated Moving Average with eXogenous variables
    # 即"季节性自回归积分滑动平均模型"，是经典的时间序列预测方法
    from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Matplotlib 绘图库导入处理 ---
# 尝试导入 matplotlib 用于生成分解图和汇总图表
# 如果不可用也不影响核心预测功能，程序会继续运行
MATPLOTLIB_AVAILABLE = False  # 标记 matplotlib 是否可用的全局标志变量
try:
    import matplotlib  # matplotlib 主模块
    # 设置后端为 'Agg'，适用于无图形界面的服务器环境（不弹出显示窗口）
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # matplotlib 的绘图接口模块
    MATPLOTLIB_AVAILABLE = True  # 标记 matplotlib 可用
    print("[信息] 成功加载 matplotlib，将生成分解图。")
except ImportError:
    # 当 matplotlib 未安装时，输出警告但继续执行
    print("[警告] matplotlib 未安装，跳过图表生成。")

# 忽略所有运行时警告，保持控制台输出整洁
warnings.filterwarnings('ignore')

# ============================================================
# 第二部分: 配置参数
# ============================================================

# --- 路径配置 ---
# 定义数据文件和输出目录的路径
# 使用相对路径，假设脚本在项目的根目录下运行
DATA_DIR = 'data'  # 输入数据文件存放目录
RESULTS_DIR = 'results'  # 结果输出目录

# --- 时间范围配置 ---
# 定义训练集的时间范围（模型学习的历史数据区间）
TRAIN_START = '2020-01-01'  # 训练开始日期，使用2020年初作为起点
TRAIN_END = '2024-12-31'  # 训练结束日期，使用2024年底作为终点

# 定义未来预测的时间范围
FORECAST_START = '2026-06-01'  # 预测开始日期（需要预测的月份开始日期）
FORECAST_END = '2026-08-30'  # 预测结束日期（需要预测的月份结束日期）
FORECAST_DAYS = 90  # 预测天数（90天）

# --- SKU 筛选配置 ---
# 只处理 demand_class 为 'seasonal' 的 SKU（具有季节性特征的药品）
TARGET_DEMAND_CLASS = 'seasonal'  # 目标需求类别：季节性药品

# --- 渠道定义 ---
# 定义4个需要分别预测的渠道列名
# 这4列在 demand_daily.csv 中分别对应不同销售渠道的销量数据
CHANNEL_COLUMNS = {
    'total': 'demand_total',  # 总销量：所有渠道的销量之和
    'hospital': 'demand_hospital',  # 医院渠道销量：通过医院销售的药品数量
    'chain': 'demand_chain',  # 连锁药店渠道销量：通过连锁药店销售的药品数量
    'independent': 'demand_independent',  # 独立药店渠道销量：通过独立药店销售的药品数量
}

# 渠道的中文名称映射，用于输出和日志显示
CHANNEL_NAMES = {
    'total': '总销量',  # total 渠道的中文名称
    'hospital': '医院渠道',  # hospital 渠道的中文名称
    'chain': '连锁药店渠道',  # chain 渠道的中文名称
    'independent': '独立药店渠道',  # independent 渠道的中文名称
}

# --- 外部回归量配置 ---
# 定义使用的外部回归量列名（从 external_signals.csv 中读取）
# 这些变量作为额外输入特征帮助模型理解需求变化的外部驱动因素
EXOGENOUS_FEATURES = [
    'flu_activity_index',  # 流感活动指数：反映流感严重程度，影响药品需求
    'is_holiday',  # 是否假期：假期期间药品需求模式不同
    'is_weekend',  # 是否周末：周末医院/药房运营模式不同导致需求变化
]

# --- Prophet 模型参数配置 ---
# Prophet 模型的超参数设置
PROPHET_YEARLY = True  # 启用年季节性：Prophet 自动拟合年度周期模式（如流感季）
PROPHET_WEEKLY = True  # 启用周季节性：Prophet 自动拟合每周周期模式（如周末效应）
PROPHET_DAILY = False  # 禁用日季节性：日度数据不需要日级别周期
PROPHET_CHANGEPOINT_SCALE = 0.05  # 趋势变化点先验尺度：越小表示趋势越保守、越稳定
PROPHET_SEASONALITY_SCALE = 10.0  # 季节性先验尺度：越大表示季节性效应越明显

# ============================================================
# 第三部分: 辅助函数
# ============================================================


def ensure_dir(directory):
    """
    确保目录存在，如果不存在则自动创建。

    参数:
        directory (str): 需要确保存在的目录路径

    功能:
        使用 os.makedirs 递归创建目录，exist_ok=True 表示目录已存在时不抛出异常
    """
    # exist_ok=True 是关键参数，确保目录已存在时不会报错
    os.makedirs(directory, exist_ok=True)


def calculate_mape(y_true, y_pred):
    """
    计算 MAPE (Mean Absolute Percentage Error，平均绝对百分比误差)

    公式: MAPE = mean(|(实际值 - 预测值) / 实际值|) * 100%

    参数:
        y_true (array-like): 实际观测值数组
        y_pred (array-like): 模型预测值数组

    返回:
        float: MAPE 值（百分比形式），值越小表示预测越准确

    注意:
        当实际值为0时直接返回 100% 以避免除零错误
    """
    # 将输入转换为 numpy 数组并指定为浮点类型，便于向量化计算
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # 创建布尔掩码：过滤掉实际值为0的数据点，避免除零错误
    mask = y_true != 0  # mask 是布尔数组，True 表示该位置实际值不为0

    # 如果所有实际值都为0，直接返回 100%（最坏情况的预测误差）
    if mask.sum() == 0:
        return 100.0

    # 计算每个数据点的绝对百分比误差：abs((实际值 - 预测值) / 实际值)
    absolute_percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])

    # 计算所有数据点的平均绝对百分比误差，并转换为百分比形式
    mape = np.mean(absolute_percentage_errors) * 100

    return mape


def calculate_rmse(y_true, y_pred):
    """
    计算 RMSE (Root Mean Squared Error，均方根误差)

    公式: RMSE = sqrt(mean((实际值 - 预测值)^2))

    参数:
        y_true (array-like): 实际观测值数组
        y_pred (array-like): 模型预测值数组

    返回:
        float: RMSE 值，与原始数据同单位，值越小表示预测越准确
        RMSE 对大误差更敏感（因为平方操作放大了大的偏差）
    """
    # 将输入转换为 numpy 数组并指定为浮点类型
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # 计算均方误差: mean((实际值 - 预测值)^2)
    mse = np.mean((y_true - y_pred) ** 2)

    # 取平方根得到 RMSE，使单位与原始数据一致
    rmse = np.sqrt(mse)

    return rmse


# ============================================================
# 第四部分: 数据加载函数
# ============================================================


def load_data():
    """
    加载所有需要的 CSV 数据文件。

    加载的文件:
        1. demand_daily.csv - 日度需求数据（包含4个渠道的销量列）
        2. sku_profiles.csv - SKU 画像信息（分类、基准需求等）
        3. external_signals.csv - 外部信号（流感指数、假期、周末等）
        4. holidays.csv - 假期日历（用于 Prophet 假期效应建模）

    返回:
        tuple: (demand_df, sku_df, external_df, holidays_df) 四个 DataFrame

    异常:
        如果关键文件不存在，打印错误信息并退出程序
    """
    # 打印数据加载阶段的标题信息
    print("\n" + "=" * 60)
    print("[步骤 1/5] 正在加载数据文件...")
    print("=" * 60)

    # --- 4.1 加载日度需求数据 ---
    # demand_daily.csv 包含每个 SKU 每天的各类需求数据（包括4个渠道）
    demand_path = os.path.join(DATA_DIR, 'demand_daily.csv')
    print(f"[加载] 正在读取需求数据: {demand_path}")

    # 使用 pandas 的 read_csv 函数读取 CSV 文件
    # parse_dates=['date'] 参数将 date 列自动解析为日期时间格式
    demand_df = pd.read_csv(demand_path, parse_dates=['date'])

    # 打印需求数据的基本信息，帮助理解数据结构和规模
    print(f"[信息] 需求数据: {len(demand_df)} 行, {demand_df.shape[1]} 列")
    print(f"[信息] 时间范围: {demand_df['date'].min()} ~ {demand_df['date'].max()}")
    print(f"[信息] SKU数量: {demand_df['sku_id'].nunique()}")
    print(f"[信息] 列名: {list(demand_df.columns)}")

    # --- 4.2 加载 SKU 画像数据 ---
    # sku_profiles.csv 包含每个 SKU 的特征和分类信息（如 demand_class）
    sku_path = os.path.join(DATA_DIR, 'sku_profiles.csv')
    print(f"\n[加载] 正在读取SKU画像: {sku_path}")

    sku_df = pd.read_csv(sku_path)

    print(f"[信息] SKU画像: {len(sku_df)} 行, {sku_df.shape[1]} 列")
    print(f"[信息] demand_class 分布:\n{sku_df['demand_class'].value_counts()}")

    # --- 4.3 加载外部信号数据 ---
    # external_signals.csv 包含各种可能影响药品需求的外部因素
    external_path = os.path.join(DATA_DIR, 'external_signals.csv')
    print(f"\n[加载] 正在读取外部信号: {external_path}")

    external_df = pd.read_csv(external_path, parse_dates=['date'])

    print(f"[信息] 外部信号: {len(external_df)} 行, {external_df.shape[1]} 列")
    print(f"[信息] 列名: {list(external_df.columns)}")

    # --- 4.4 加载假期日历 ---
    # holidays.csv 包含节假日信息，用于 Prophet 的假期效应建模
    holidays_path = os.path.join(DATA_DIR, 'holidays.csv')
    print(f"\n[加载] 正在读取假期日历: {holidays_path}")

    holidays_df = pd.read_csv(holidays_path, parse_dates=['date'])

    print(f"[信息] 假期日历: {len(holidays_df)} 行")
    print(f"[信息] 假期列表:\n{holidays_df['holiday_name'].value_counts().head(10)}")

    # 打印数据加载完成的信息
    print("\n[完成] 所有数据文件加载成功!")

    # 返回四个 DataFrame 组成的元组
    return demand_df, sku_df, external_df, holidays_df


# ============================================================
# 第五部分: 数据预处理函数
# ============================================================


def prepare_prophet_holidays(holidays_df):
    """
    将假期数据转换为 Prophet 需要的格式。

    Prophet 要求假期数据的格式为 DataFrame，包含以下列:
        - 'holiday': 假期名称（字符串类型）
        - 'ds': 假期日期（日期时间格式，Prophet 使用 'ds' 表示日期）
        - 'lower_window': 假期效应开始前的天数（负数表示提前）
        - 'upper_window': 假期效应结束后的天数（正数表示延后）

    参数:
        holidays_df (pd.DataFrame): 原始假期数据，包含 date, holiday_name 等列

    返回:
        pd.DataFrame: Prophet 格式的假期数据
    """
    print("[预处理] 正在准备 Prophet 假期数据...")

    # 创建 Prophet 格式的假期 DataFrame
    # Prophet 要求列名为 'holiday', 'ds', 'lower_window', 'upper_window'
    prophet_holidays = pd.DataFrame({
        'holiday': holidays_df['holiday_name'],  # 假期名称列
        'ds': holidays_df['date'],  # 假期日期列（Prophet 使用 'ds' 作为日期列名）
        'lower_window': -1,  # 假期效应提前1天开始（如节前备货增加需求）
        'upper_window': 1,  # 假期效应延后1天结束（如节后补货增加需求）
    })

    print(f"[信息] 假期数据准备完成: {len(prophet_holidays)} 个假期记录")

    return prophet_holidays


def prepare_training_data(sku_id, channel_key, demand_df, external_df):
    """
    为单个 SKU 的单个渠道准备训练数据，构建 Prophet 需要的格式。

    Prophet 需要的输入格式:
        - 'ds' 列: 日期（datetime 类型）
        - 'y' 列: 目标值（该渠道的销量）
        - 额外的列: 外部回归量（作为预测时的额外输入特征）

    参数:
        sku_id (str/int): SKU 的唯一标识符
        channel_key (str): 渠道标识符（'total', 'hospital', 'chain', 'independent'）
        demand_df (pd.DataFrame): 日度需求数据
        external_df (pd.DataFrame): 外部信号数据

    返回:
        pd.DataFrame: 准备好的训练数据，符合 Prophet 格式要求
    """
    # --- 5.1 筛选当前 SKU 的需求数据 ---
    # 从 demand_df 中筛选出当前 SKU 的所有行
    sku_demand = demand_df[demand_df['sku_id'] == sku_id].copy()

    # 按日期排序，确保时间序列按正确的时间顺序排列
    sku_demand = sku_demand.sort_values('date').reset_index(drop=True)

    # --- 5.2 筛选训练时间范围内的数据 ---
    # 将训练开始和结束日期转换为 datetime 对象，便于日期比较
    train_start_dt = pd.to_datetime(TRAIN_START)  # 训练开始日期转换为 datetime
    train_end_dt = pd.to_datetime(TRAIN_END)  # 训练结束日期转换为 datetime

    # 使用布尔索引筛选训练集：日期在 TRAIN_START 和 TRAIN_END 之间（包含边界）
    train_mask = (sku_demand['date'] >= train_start_dt) & (sku_demand['date'] <= train_end_dt)
    train_data = sku_demand.loc[train_mask].copy()  # 复制避免 SettingWithCopy 警告

    # --- 5.3 构建 Prophet 格式 ---
    # Prophet 要求列名为 'ds'（日期）和 'y'（目标值）
    # 获取当前渠道对应的列名
    target_col = CHANNEL_COLUMNS[channel_key]  # 根据渠道标识获取对应的数据列名

    prophet_df = pd.DataFrame({
        'ds': train_data['date'],  # 'ds' = datestamp，Prophet 的日期列
        'y': train_data[target_col],  # 'y' = 目标变量（当前渠道的销量）
    })

    # --- 5.4 合并外部回归量 ---
    # 从 external_df 中筛选训练时间范围内的外部信号数据
    ext_mask = (external_df['date'] >= train_start_dt) & (external_df['date'] <= train_end_dt)
    ext_train = external_df.loc[ext_mask].copy()

    # 将外部信号按日期合并到 prophet_df 中
    # 使用 merge 函数按日期列进行左连接（保留所有 prophet_df 的行）
    prophet_df = prophet_df.merge(
        ext_train[['date'] + EXOGENOUS_FEATURES],  # 只选择日期列和需要的外部特征列
        left_on='ds',  # prophet_df 的日期列名
        right_on='date',  # external_df 的日期列名
        how='left'  # 左连接，保留所有训练日期（即使外部信号缺失）
    )

    # 删除合并后的冗余 'date' 列（因为已经有 'ds' 列了）
    if 'date' in prophet_df.columns:
        prophet_df = prophet_df.drop(columns=['date'])

    # --- 5.5 处理缺失值 ---
    # 外部回归量中可能存在缺失值，用 0 填充（假设缺失意味着该效应不存在）
    for feature in EXOGENOUS_FEATURES:
        if feature in prophet_df.columns:
            # fillna(0) 将缺失值填充为 0
            prophet_df[feature] = prophet_df[feature].fillna(0)

    # --- 5.6 处理目标变量的缺失值和负值 ---
    # 删除目标值为 NaN 的行（无法用于训练，因为 Prophet 需要完整的 y 值）
    prophet_df = prophet_df.dropna(subset=['y'])

    # 将负值替换为 0（药品销量不应为负数，负值可能是数据错误）
    prophet_df['y'] = prophet_df['y'].clip(lower=0)

    return prophet_df


def prepare_future_dataframe(external_df):
    """
    为未来预测时间段创建 Prophet 格式的 DataFrame。

    预测时间范围: 2026-06-01 至 2026-06-30（30天）

    参数:
        external_df (pd.DataFrame): 外部信号数据

    返回:
        pd.DataFrame: 未来日期的 DataFrame，包含 'ds' 和外部回归量列
    """
    # 创建预测时间段的日期范围
    # pd.date_range 生成从 FORECAST_START 到 FORECAST_END 的每日日期序列
    future_dates = pd.date_range(
        start=FORECAST_START,  # 预测开始日期
        end=FORECAST_END,  # 预测结束日期
        freq='D'  # 日度频率（每天一个数据点）
    )

    # 创建 Prophet 格式的未来日期 DataFrame
    future_df = pd.DataFrame({'ds': future_dates})

    # --- 合并外部回归量到未来日期 ---
    # 需要为 future_df 中的每个日期提供外部回归量的值
    for feature in EXOGENOUS_FEATURES:
        if feature in external_df.columns:
            # 创建一个日期到特征值的映射字典（使用 set_index 建立索引）
            feature_map = external_df.set_index('date')[feature]
            # 将特征值映射到 future_df 的日期（通过 map 方法查找对应值）
            future_df[feature] = future_df['ds'].map(feature_map)
            # 对于没有外部信号的日期，用 0 填充（假设效应不存在）
            future_df[feature] = future_df[feature].fillna(0)

    return future_df


def prepare_validation_data(sku_id, channel_key, demand_df, external_df):
    """
    为单个 SKU 的单个渠道准备验证数据，用于评估模型性能。

    验证集使用时间序列的后段数据来评估模型预测精度。

    参数:
        sku_id (str/int): SKU 的唯一标识符
        channel_key (str): 渠道标识符（'total', 'hospital', 'chain', 'independent'）
        demand_df (pd.DataFrame): 日度需求数据
        external_df (pd.DataFrame): 外部信号数据

    返回:
        tuple: (val_prophet_df, target_col)
            - val_prophet_df: 验证集 Prophet 格式 DataFrame
            - target_col: 目标列名
    """
    # 筛选当前 SKU 的需求数据
    sku_demand = demand_df[demand_df['sku_id'] == sku_id].copy()
    sku_demand = sku_demand.sort_values('date')  # 按日期排序

    # 获取当前渠道对应的列名
    target_col = CHANNEL_COLUMNS[channel_key]

    # 筛选验证时间范围：使用训练结束后、预测开始前的数据作为验证集
    # 与其他模型保持一致：验证集为 TRAIN_END 次日 至 FORECAST_START 前一日
    val_start_dt = pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1)  # 验证开始日期
    val_end_dt = pd.to_datetime(FORECAST_START) - pd.Timedelta(days=1)  # 验证结束日期

    # 布尔索引筛选验证时间范围
    val_mask = (sku_demand['date'] >= val_start_dt) & (sku_demand['date'] <= val_end_dt)
    val_data = sku_demand.loc[val_mask].copy()

    # 如果验证数据不足，返回 None
    if len(val_data) < 7:  # 至少需要7天验证数据
        return None, target_col

    # 构建 Prophet 格式的验证数据
    val_prophet_df = pd.DataFrame({
        'ds': val_data['date'],  # 日期列
        'y': val_data[target_col],  # 目标值（当前渠道的实际销量）
    })

    # 合并外部回归量到验证数据
    for feature in EXOGENOUS_FEATURES:
        if feature in external_df.columns:
            feature_map = external_df.set_index('date')[feature]
            val_prophet_df[feature] = val_prophet_df['ds'].map(feature_map)
            val_prophet_df[feature] = val_prophet_df[feature].fillna(0)

    # 处理缺失值和负值
    val_prophet_df = val_prophet_df.dropna(subset=['y'])
    val_prophet_df['y'] = val_prophet_df['y'].clip(lower=0)

    return val_prophet_df, target_col


# ============================================================
# 第六部分: 模型训练与预测函数
# ============================================================


def train_prophet_model(train_df, prophet_holidays):
    """
    使用 Prophet 训练时间序列预测模型。

    Prophet 模型特点:
        - 自动拟合趋势（线性或 logistic 增长）
        - 自动检测年、周、日季节性
        - 支持假期效应建模
        - 支持外部回归量（额外特征输入）

    参数:
        train_df (pd.DataFrame): 训练数据，包含 ds, y 和外部回归量列
        prophet_holidays (pd.DataFrame): Prophet 格式的假期数据

    返回:
        Prophet: 训练好的 Prophet 模型对象
    """
    # --- 6.1 初始化 Prophet 模型 ---
    # 创建 Prophet 实例，传入配置参数
    model = Prophet(
        yearly_seasonality=PROPHET_YEARLY,  # 启用年季节性（自动拟合年度周期如流感季）
        weekly_seasonality=PROPHET_WEEKLY,  # 启用周季节性（自动拟合每周模式如周末效应）
        daily_seasonality=PROPHET_DAILY,  # 禁用日季节性（日度数据不需要日级别周期）
        holidays=prophet_holidays,  # 传入假期数据用于建模假期效应
        seasonality_mode='multiplicative',  # 季节性模式：乘法（季节性幅度随趋势增长）
        changepoint_prior_scale=PROPHET_CHANGEPOINT_SCALE,  # 趋势变化点灵活性（越小越保守）
        seasonality_prior_scale=PROPHET_SEASONALITY_SCALE,  # 季节性灵活性（越大越明显）
        interval_width=0.8,  # 预测区间的置信度（80% 置信区间）
    )

    # --- 6.2 添加外部回归量 ---
    # 遍历所有外部特征列，逐一添加到 Prophet 模型中
    for feature in EXOGENOUS_FEATURES:
        if feature in train_df.columns:
            # add_regressor 方法将列注册为外部回归量
            # standardize=True 表示对回归量进行标准化（均值为0，标准差为1）
            model.add_regressor(name=feature, standardize=True)

    # --- 6.3 拟合模型 ---
    # fit 方法训练模型，让 Prophet 学习数据中的趋势、季节性和外部效应
    model.fit(train_df)

    return model


def train_sarimax_model(train_df):
    """
    使用 SARIMAX 作为 Prophet 的备选模型。

    SARIMAX(p,d,q)(P,D,Q)m 参数说明:
        - (p,d,q): 非季节性 ARIMA 参数（自回归阶数、差分阶数、移动平均阶数）
        - (P,D,Q): 季节性 ARIMA 参数
        - m: 季节周期（7 表示周周期）

    参数:
        train_df (pd.DataFrame): 训练数据，包含 ds, y 和外部回归量列

    返回:
        tuple: (fitted_model, model_params)
            - fitted_model: 训练好的 SARIMAX 模型
            - model_params: 包含模型元信息的字典
    """
    # --- 6.S1 准备数据 ---
    # SARIMAX 需要设置日期索引
    y_series = train_df.set_index('ds')['y']  # 目标变量时间序列，以日期为索引

    # 提取外部回归量（在训练数据中存在的外部特征列）
    exog_cols = [f for f in EXOGENOUS_FEATURES if f in train_df.columns]
    if exog_cols:
        # 从训练数据中提取外部回归量并设置日期索引
        exog_df = train_df.set_index('ds')[exog_cols]
        # 填充缺失值为 0（假设缺失意味着该效应不存在）
        exog_df = exog_df.fillna(0)
    else:
        # 如果没有外部回归量，设为 None
        exog_df = None

    # --- 6.S2 创建并训练 SARIMAX 模型 ---
    # 使用 (1,1,1)(1,1,1,7) 参数配置:
    #   order=(1,1,1): 非季节部分 - 1阶自回归，1阶差分，1阶移动平均
    #   seasonal_order=(1,1,1,7): 季节部分 - 1阶季节自回归，1阶季节差分，1阶季节移动平均，周期为7天（周）
    print("[SARIMAX] 使用 SARIMAX(1,1,1)(1,1,1,7) 模型...")

    model = SARIMAX(
        y_series,  # 目标时间序列数据
        exog=exog_df,  # 外部回归量（可选）
        order=(1, 1, 1),  # 非季节 ARIMA 参数 (p,d,q)
        seasonal_order=(1, 1, 1, 7),  # 季节 ARIMA 参数 (P,D,Q,s)，季节周期=7天
        enforce_stationarity=False,  # 不强制平稳性（实际药品需求数据往往不平稳）
        enforce_invertibility=False,  # 不强制可逆性
    )

    # fit 方法使用最大似然估计训练模型
    # disp=False 关闭迭代过程中的输出信息，保持控制台整洁
    fitted_model = model.fit(disp=False)

    # 保存模型参数信息供后续预测使用
    model_params = {
        'exog_cols': exog_cols,  # 使用的外部回归量列名列表
    }

    return fitted_model, model_params


def generate_forecast_prophet(model, future_df):
    """
    使用训练好的 Prophet 模型生成未来预测。

    参数:
        model (Prophet): 训练好的 Prophet 模型对象
        future_df (pd.DataFrame): 未来日期的 DataFrame，包含 ds 和外部回归量

    返回:
        pd.DataFrame: 预测结果，包含预测值、趋势、季节性分解等
    """
    # Prophet 的 predict 方法生成预测结果
    # 返回的 DataFrame 包含以下列:
    #   ds: 日期
    #   yhat: 预测值（点估计）
    #   yhat_lower: 预测区间下限（80%置信区间）
    #   yhat_upper: 预测区间上限（80%置信区间）
    #   trend: 趋势分量
    #   yearly: 年季节性分量
    #   weekly: 周季节性分量
    forecast = model.predict(future_df)

    return forecast


def generate_forecast_sarimax(fitted_model, model_params, future_df):
    """
    使用训练好的 SARIMAX 模型生成未来预测。

    参数:
        fitted_model: 训练好的 SARIMAX 模型对象
        model_params (dict): 包含模型元信息的字典
        future_df (pd.DataFrame): 未来日期的 DataFrame，包含 ds 和外部回归量

    返回:
        pd.DataFrame: 预测结果，格式与 Prophet 输出兼容
    """
    # 提取外部回归量列名
    exog_cols = model_params['exog_cols']

    # 准备未来外部回归量
    if exog_cols and exog_cols[0] in future_df.columns:
        # 从 future_df 中提取外部回归量并填充缺失值
        exog_future = future_df[exog_cols].fillna(0)
    else:
        # 如果没有外部回归量，设为 None
        exog_future = None

    # get_forecast 方法生成预测
    # steps: 预测步数（等于未来日期数量）
    forecast_result = fitted_model.get_forecast(
        steps=len(future_df),  # 预测步数等于未来数据行数
        exog=exog_future  # 外部回归量
    )

    # 提取预测值和置信区间
    forecast_mean = forecast_result.predicted_mean  # 点预测值
    conf_int = forecast_result.conf_int()  # 置信区间

    # 构建与 Prophet 输出格式兼容的 DataFrame
    forecast_df = pd.DataFrame({
        'ds': future_df['ds'].values,  # 日期列
        'yhat': forecast_mean.values,  # 预测值
        'yhat_lower': conf_int.iloc[:, 0].values,  # 置信区间下限
        'yhat_upper': conf_int.iloc[:, 1].values,  # 置信区间上限
    })

    # 将负值预测截断为 0（药品需求不能为负数）
    forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
    forecast_df['yhat_lower'] = forecast_df['yhat_lower'].clip(lower=0)

    return forecast_df


# ============================================================
# 第七部分: 主流程函数
# ============================================================


def run_prophet_pipeline():
    """
    运行完整的 Prophet 预测流程，对4个渠道分别训练和预测。

    流程步骤:
        1. 加载所有数据文件
        2. 筛选 seasonal 类型的 SKU
        3. 对每个 SKU:
           a. 对4个渠道分别准备数据
           b. 对4个渠道分别训练模型
           c. 对4个渠道分别生成预测
           d. 对4个渠道分别评估模型性能
        4. 汇总所有结果并保存为 CSV 文件

    返回:
        tuple: (predictions_df, metrics_df) 预测结果和评估指标的 DataFrame
    """
    # --- 7.1 创建输出目录 ---
    # 确保结果输出目录存在（如果不存在则自动创建）
    ensure_dir(RESULTS_DIR)

    # --- 7.2 加载所有数据 ---
    # 调用 load_data 函数加载4个数据文件
    demand_df, sku_df, external_df, holidays_df = load_data()

    print("\n" + "=" * 60)
    print("[步骤 2/5] 正在筛选 seasonal 类型SKU...")
    print("=" * 60)

    # --- 7.3 筛选 seasonal 类型的 SKU ---
    # 从 sku_df 中筛选 demand_class 为 'seasonal' 的 SKU 列表
    seasonal_skus = sku_df[sku_df['demand_class'] == TARGET_DEMAND_CLASS]['sku_id'].tolist()

    print(f"[信息] 找到 {len(seasonal_skus)} 个 seasonal 类型的SKU")

    # 如果 seasonal SKU 数量太多，限制处理数量（防止运行时间过长）
    MAX_SKUS = 500  # 最大处理 SKU 数量
    if len(seasonal_skus) > MAX_SKUS:
        print(f"[信息] SKU数量超过 {MAX_SKUS}，只处理前 {MAX_SKUS} 个")
        seasonal_skus = seasonal_skus[:MAX_SKUS]

    # --- 7.4 准备假期数据（仅用于 Prophet 模型）---
    if PROPHET_AVAILABLE:
        # Prophet 可用时，准备 Prophet 格式的假期数据
        prophet_holidays = prepare_prophet_holidays(holidays_df)
    else:
        # Prophet 不可用时，假期数据设为 None（SARIMAX 不使用假期数据）
        prophet_holidays = None

    # --- 7.5 初始化结果存储列表 ---
    all_predictions = []  # 存储所有 SKU 的所有渠道预测结果
    all_metrics = []  # 存储所有 SKU 的所有渠道评估指标

    print("\n" + "=" * 60)
    print(f"[步骤 3/5] 开始训练模型 ({'Prophet' if PROPHET_AVAILABLE else 'SARIMAX (备选)'})...")
    print(f"[信息] 将对4个渠道分别训练: {list(CHANNEL_NAMES.values())}")
    print("=" * 60)

    # --- 7.6 遍历每个 seasonal SKU 进行训练和预测 ---
    for idx, sku_id in enumerate(seasonal_skus):
        # 打印进度信息，每处理 10 个 SKU 输出一次进度
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"\n[进度] 正在处理第 {idx + 1}/{len(seasonal_skus)} 个SKU: {sku_id}")

        try:
            # 初始化当前 SKU 的4个渠道预测结果字典
            # 键为渠道标识，值为该渠道的预测 DataFrame
            sku_forecasts = {}

            # 初始化当前 SKU 的4个渠道验证集预测值字典（用于评估）
            sku_val_predictions = {}
            sku_val_actuals = {}

            # ====== 对每个渠道的循环 ======
            # 遍历4个渠道：total, hospital, chain, independent
            for channel_key in CHANNEL_COLUMNS.keys():
                channel_name = CHANNEL_NAMES[channel_key]  # 获取渠道中文名称

                # --- 7.6.1 准备当前渠道的训练数据 ---
                train_df = prepare_training_data(sku_id, channel_key, demand_df, external_df)

                # 检查训练数据是否足够（Prophet 要求至少2个周期的数据）
                if len(train_df) < 14:  # 至少需要14天数据才能训练有意义的模型
                    print(f"[跳过] SKU {sku_id} 渠道 {channel_name} 训练数据不足 ({len(train_df)} 行)")
                    continue

                # --- 7.6.2 训练模型 ---
                if PROPHET_AVAILABLE:
                    # 使用 Prophet 训练当前渠道的模型
                    model = train_prophet_model(train_df, prophet_holidays)
                else:
                    # 使用 SARIMAX 作为备选训练当前渠道的模型
                    fitted_model, model_params = train_sarimax_model(train_df)

                # --- 7.6.3 生成未来预测 ---
                # 为未来预测时间段创建 DataFrame（包含外部回归量）
                future_df = prepare_future_dataframe(external_df)

                if PROPHET_AVAILABLE:
                    # --- Prophet 预测方式 ---
                    # Prophet 需要创建包含历史日期和未来日期的完整 DataFrame
                    # 计算从训练结束到预测结束所需的天数，确保覆盖到 FORECAST_END
                    train_end_dt = pd.to_datetime(TRAIN_END)
                    forecast_end_dt = pd.to_datetime(FORECAST_END)
                    periods_needed = (forecast_end_dt - train_end_dt).days
                    full_future_df = model.make_future_dataframe(periods=periods_needed)

                    # 将外部回归量合并到完整未来 DataFrame
                    for feature in EXOGENOUS_FEATURES:
                        if feature in external_df.columns:
                            # 创建日期到特征值的映射
                            feature_map = external_df.set_index('date')[feature]
                            # 将特征值映射到 full_future_df 的日期
                            full_future_df[feature] = full_future_df['ds'].map(feature_map)
                            # 对于没有外部信号的日期，用 0 填充
                            full_future_df[feature] = full_future_df[feature].fillna(0)

                    # 使用 Prophet 模型生成预测
                    forecast = generate_forecast_prophet(model, full_future_df)

                    # 提取预测时间段的结果（只保留未来日期的预测）
                    forecast_future = forecast[forecast['ds'] >= FORECAST_START].copy()

                else:
                    # --- SARIMAX 预测方式 ---
                    # 使用 SARIMAX 模型生成未来预测
                    forecast_future = generate_forecast_sarimax(
                        fitted_model, model_params, future_df
                    )

                # 将当前渠道的预测结果保存到字典中
                sku_forecasts[channel_key] = forecast_future

                # --- 7.6.4 在验证集上评估当前渠道 ---
                val_df, _ = prepare_validation_data(sku_id, channel_key, demand_df, external_df)

                if val_df is not None and len(val_df) > 0:
                    # 验证数据存在，进行验证集预测并计算评估指标
                    if PROPHET_AVAILABLE:
                        # Prophet 验证集预测
                        # 使用 predict 方法在验证集上预测
                        val_forecast = model.predict(val_df)
                        y_pred = val_forecast['yhat'].values  # 预测值
                    else:
                        # SARIMAX 验证集预测
                        # 准备验证集外部回归量
                        val_exog_cols = model_params['exog_cols']
                        if val_exog_cols and val_exog_cols[0] in val_df.columns:
                            val_exog = val_df[val_exog_cols].fillna(0)
                        else:
                            val_exog = None

                        # 使用 get_prediction 进行验证集预测
                        val_forecast_result = fitted_model.get_prediction(
                            start=0,  # 从验证集开始
                            end=len(val_df) - 1,  # 到验证集结束
                            exog=val_exog  # 验证集外部回归量
                        )
                        y_pred = val_forecast_result.predicted_mean.values

                    # 获取验证集实际值
                    y_true = val_df['y'].values

                    # 截断负值预测为 0（需求不能为负）
                    y_pred = np.clip(y_pred, 0, None)

                    # 计算评估指标：MAPE 和 RMSE
                    mape = calculate_mape(y_true, y_pred)
                    rmse = calculate_rmse(y_true, y_pred)
                else:
                    # 如果验证数据不足，指标设为 NaN（表示无法评估）
                    mape = np.nan
                    rmse = np.nan

                # --- 7.6.5 保存当前渠道的预测结果 ---
                # 遍历当前渠道的每个预测日期，创建结果记录
                if channel_key in sku_forecasts:
                    forecast_future = sku_forecasts[channel_key]
                    for _, row in forecast_future.iterrows():
                        all_predictions.append({
                            'sku_id': sku_id,  # SKU 唯一标识
                            'channel': channel_key,  # 渠道标识
                            'date': row['ds'],  # 预测日期
                            'forecast': row['yhat'],  # 预测值
                            'forecast_lower': row.get('yhat_lower', np.nan),  # 预测区间下限
                            'forecast_upper': row.get('yhat_upper', np.nan),  # 预测区间上限
                        })

                # --- 7.6.6 保存当前渠道的评估指标 ---
                all_metrics.append({
                    'sku_id': sku_id,  # SKU 唯一标识
                    'channel': channel_key,  # 渠道标识
                    'train_samples': len(train_df),  # 训练样本数量
                    'mape': mape,  # MAPE 评估指标
                    'rmse': rmse,  # RMSE 评估指标
                })

                # --- 7.6.7 生成分解图（仅为前3个SKU的每个渠道生成，避免图片过多）---
                if MATPLOTLIB_AVAILABLE and PROPHET_AVAILABLE and idx < 3:
                    plot_decomposition(model, forecast, sku_id, channel_key)

            # --- 7.6.8 打印当前 SKU 的处理完成信息 ---
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[完成] SKU {sku_id} 的4个渠道全部处理完成")

        except Exception as e:
            # 捕获任何异常，打印错误信息但继续处理下一个 SKU（保证流程不中断）
            print(f"[错误] 处理SKU {sku_id} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细的异常堆栈信息便于调试
            continue

    # --- 7.7 汇总并保存结果 ---
    print("\n" + "=" * 60)
    print("[步骤 4/5] 正在保存结果...")
    print("=" * 60)

    # ====== 7.7.1 保存预测结果 ======
    # 将所有预测结果转换为 DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    if len(predictions_df) > 0:
        # --- 将预测结果从长格式转换为宽格式（4列渠道预测值）---
        # 使用 pivot_table 将每个渠道的预测值展开为单独的列
        pivot_predictions = predictions_df.pivot_table(
            index=['sku_id', 'date'],  # 以 SKU 和日期为索引
            columns='channel',  # 以渠道为列
            values='forecast',  # 以预测值为数据
            aggfunc='first'  # 聚合函数（每个 SKU+日期+渠道只有一条记录）
        ).reset_index()

        # 重命名列：将渠道预测值列改为标准名称
        pivot_predictions.columns.name = None  # 移除列名
        # 确保4个渠道的列都存在（如果某个渠道缺失则创建空列）
        for ch_key in CHANNEL_COLUMNS.keys():
            if ch_key not in pivot_predictions.columns:
                pivot_predictions[ch_key] = np.nan

        # 重命名列以匹配要求的输出格式
        pivot_predictions = pivot_predictions.rename(columns={
            'total': 'forecast_total',
            'hospital': 'forecast_hospital',
            'chain': 'forecast_chain',
            'independent': 'forecast_independent',
        })

        # 确保输出列顺序正确
        output_cols = ['sku_id', 'date', 'forecast_total', 'forecast_hospital',
                       'forecast_chain', 'forecast_independent']
        # 只保留存在的列
        output_cols = [c for c in output_cols if c in pivot_predictions.columns]
        pivot_predictions = pivot_predictions[output_cols]

        # 将日期列格式化为字符串
        pivot_predictions['date'] = pd.to_datetime(pivot_predictions['date']).dt.strftime('%Y-%m-%d')

        # 保存预测结果为 CSV 文件
        predictions_path = os.path.join(RESULTS_DIR, 'prophet_predictions.csv')
        pivot_predictions.to_csv(predictions_path, index=False)
        print(f"[保存] 预测结果已保存到: {predictions_path}")
        print(f"[信息] 预测结果: {len(pivot_predictions)} 行")
        print(f"[信息] 列: {list(pivot_predictions.columns)}")
    else:
        # 如果没有预测结果，创建一个空的 DataFrame
        pivot_predictions = pd.DataFrame()
        print("[警告] 没有生成预测结果")

    # ====== 7.7.2 保存评估指标 ======
    # 将所有评估指标转换为 DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    if len(metrics_df) > 0:
        # 保存评估指标为 CSV 文件
        metrics_path = os.path.join(RESULTS_DIR, 'prophet_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[保存] 评估指标已保存到: {metrics_path}")
        print(f"[信息] 评估指标: {len(metrics_df)} 行")
    else:
        metrics_path = os.path.join(RESULTS_DIR, 'prophet_metrics.csv')
        print("[警告] 没有生成评估指标")

    # ====== 7.7.3 打印汇总统计 ======
    if len(metrics_df) > 0:
        print("\n" + "-" * 40)
        print("评估指标汇总（按渠道）:")
        print("-" * 40)

        # 按渠道分别计算平均 MAPE 和 RMSE
        for channel_key in CHANNEL_COLUMNS.keys():
            channel_metrics = metrics_df[metrics_df['channel'] == channel_key]
            if len(channel_metrics) > 0:
                channel_name = CHANNEL_NAMES[channel_key]
                # 排除 NaN 值后计算均值
                mean_mape = channel_metrics['mape'].mean()
                mean_rmse = channel_metrics['rmse'].mean()
                median_mape = channel_metrics['mape'].median()
                median_rmse = channel_metrics['rmse'].median()

                print(f"\n渠道: {channel_name} ({channel_key})")
                print(f"  SKU数量: {len(channel_metrics)}")
                print(f"  MAPE - 平均值: {mean_mape:.2f}%, 中位数: {median_mape:.2f}%")
                print(f"  RMSE - 平均值: {mean_rmse:.2f}, 中位数: {median_rmse:.2f}")

        print("-" * 40)

    # --- 7.8 生成汇总图表 ---
    if MATPLOTLIB_AVAILABLE and len(metrics_df) > 0:
        # 绘制各渠道的指标汇总图
        plot_metrics_summary(metrics_df)

    # 打印流程完成信息
    print("\n" + "=" * 60)
    print("[步骤 5/5] 流程完成!")
    print("=" * 60)
    print(f"输出文件:")
    if len(predictions_df) > 0:
        print(f"  - {os.path.join(RESULTS_DIR, 'prophet_predictions.csv')}")
    if len(metrics_df) > 0:
        print(f"  - {os.path.join(RESULTS_DIR, 'prophet_metrics.csv')}")
    print("=" * 60)

    # 返回预测结果和评估指标的 DataFrame
    return predictions_df, metrics_df


# ============================================================
# 第八部分: 可视化函数
# ============================================================


def plot_decomposition(model, forecast, sku_id, channel_key):
    """
    绘制 Prophet 模型的趋势/季节性分解图。

    分解图展示内容:
        - 原始数据与预测值对比
        - 趋势分量（长期变化方向）
        - 年季节性分量（每年的周期性模式，如流感季）
        - 周季节性分量（每周的周期性模式，如周末效应）

    参数:
        model (Prophet): 训练好的 Prophet 模型对象
        forecast (pd.DataFrame): 预测结果 DataFrame
        sku_id (str/int): 当前 SKU 的标识，用于文件名
        channel_key (str): 渠道标识符，用于文件名
    """
    try:
        # 创建一个新的图形，设置较大的尺寸以便清晰展示
        fig = plt.figure(figsize=(14, 10))

        # 创建 4 行 1 列的子图布局
        axes = fig.subplots(4, 1)

        channel_name = CHANNEL_NAMES[channel_key]  # 获取渠道中文名称

        # --- 子图1: 预测值 ---
        axes[0].plot(forecast['ds'], forecast['yhat'], 'b-', label='预测值', linewidth=1)
        # 如果存在置信区间，填充置信区间区域
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            axes[0].fill_between(
                forecast['ds'],
                forecast['yhat_lower'],
                forecast['yhat_upper'],
                alpha=0.2, color='blue', label='80%置信区间'
            )
        axes[0].set_title(f'SKU {sku_id} - {channel_name} - 预测结果', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('需求')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- 子图2: 趋势分量 ---
        if 'trend' in forecast.columns:
            axes[1].plot(forecast['ds'], forecast['trend'], 'g-', linewidth=1.5)
            axes[1].set_title('趋势分量 (Trend)', fontsize=11)
            axes[1].set_ylabel('趋势')
            axes[1].grid(True, alpha=0.3)

        # --- 子图3: 年季节性 ---
        if 'yearly' in forecast.columns:
            axes[2].plot(forecast['ds'], forecast['yearly'], 'r-', linewidth=1)
            axes[2].set_title('年季节性 (Yearly Seasonality)', fontsize=11)
            axes[2].set_ylabel('年效应')
            axes[2].grid(True, alpha=0.3)

        # --- 子图4: 周季节性 ---
        if 'weekly' in forecast.columns:
            axes[3].plot(forecast['ds'], forecast['weekly'], 'm-', linewidth=1)
            axes[3].set_title('周季节性 (Weekly Seasonality)', fontsize=11)
            axes[3].set_ylabel('周效应')
            axes[3].grid(True, alpha=0.3)

        # 调整子图之间的间距，避免重叠
        plt.tight_layout()

        # 保存图片到结果目录
        plot_path = os.path.join(RESULTS_DIR, f'prophet_decomposition_sku_{sku_id}_{channel_key}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图形，释放内存

        print(f"[图表] 分解图已保存: {plot_path}")

    except Exception as e:
        # 绘图失败不影响主流程，只打印警告信息
        print(f"[警告] 绘制分解图时出错: {str(e)}")


def plot_metrics_summary(metrics_df):
    """
    绘制评估指标的汇总图表（按渠道分别展示）。

    包含内容:
        - 各渠道 MAPE 分布的箱线图
        - 各渠道 RMSE 分布的箱线图

    参数:
        metrics_df (pd.DataFrame): 包含各 SKU 各渠道评估指标的 DataFrame
    """
    try:
        # 过滤掉 NaN 值，确保有有效数据用于绘图
        metrics_clean = metrics_df.dropna(subset=['mape', 'rmse'])

        if len(metrics_clean) == 0:
            print("[警告] 没有足够的有效指标数据用于绘图")
            return

        # 创建图形，2 个子图水平排列
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- 子图1: 各渠道 MAPE 分布箱线图 ---
        channel_keys = list(CHANNEL_COLUMNS.keys())
        channel_labels = [CHANNEL_NAMES[k] for k in channel_keys]

        # 为每个渠道收集 MAPE 数据
        mape_data = []
        for ch in channel_keys:
            ch_mape = metrics_clean[metrics_clean['channel'] == ch]['mape'].dropna()
            mape_data.append(ch_mape)

        # 绘制箱线图
        bp1 = axes[0].boxplot(mape_data, labels=channel_labels, patch_artist=True)
        # 为每个箱体设置不同颜色
        colors = ['steelblue', 'coral', 'lightgreen', 'plum']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        axes[0].set_title('各渠道 MAPE 分布', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('MAPE (%)')
        axes[0].grid(True, alpha=0.3)

        # --- 子图2: 各渠道 RMSE 分布箱线图 ---
        rmse_data = []
        for ch in channel_keys:
            ch_rmse = metrics_clean[metrics_clean['channel'] == ch]['rmse'].dropna()
            rmse_data.append(ch_rmse)

        # 绘制箱线图
        bp2 = axes[1].boxplot(rmse_data, labels=channel_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        axes[1].set_title('各渠道 RMSE 分布', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('RMSE')
        axes[1].grid(True, alpha=0.3)

        # 调整子图间距
        plt.tight_layout()

        # 保存汇总图
        summary_path = os.path.join(RESULTS_DIR, 'prophet_metrics_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[图表] 指标汇总图已保存: {summary_path}")

    except Exception as e:
        # 绘图失败不影响主流程
        print(f"[警告] 绘制汇总图时出错: {str(e)}")


# ============================================================
# 第九部分: 主程序入口
# ============================================================

if __name__ == '__main__':
    """
    主程序入口。

    当脚本直接运行时（不是被其他模块导入时），执行以下操作:
        1. 打印脚本信息和配置参数
        2. 运行完整的预测流程
        3. 打印完成信息和耗时统计
    """
    # 打印脚本标题信息
    print("=" * 60)
    print("Prophet 季节性预测模型 - 4渠道分别预测")
    print("=" * 60)
    print(f"模型类型: {'Prophet' if PROPHET_AVAILABLE else 'SARIMAX (Prophet备选)'}")
    print(f"目标SKU类型: {TARGET_DEMAND_CLASS}")
    print(f"预测渠道: {list(CHANNEL_NAMES.values())}")
    print(f"训练时间范围: {TRAIN_START} ~ {TRAIN_END}")
    print(f"预测时间范围: {FORECAST_START} ~ {FORECAST_END} ({FORECAST_DAYS}天)")
    print(f"外部回归量: {EXOGENOUS_FEATURES}")
    print("=" * 60)

    # 记录开始时间
    start_time = datetime.now()
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行主流程：对4个渠道分别训练和预测
    predictions, metrics = run_prophet_pipeline()

    # 记录结束时间并计算耗时
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print("\n脚本执行完毕!")
