#!/usr/bin/env python
"""
ETS指数平滑模型 — 03_model_ets.py
==================================
针对 fast 类型SKU的指数平滑预测模型，分别对4个渠道进行预测。

适用SKU特征: 销量高且稳定，波动小（demand_class = 'fast'）
模型: Exponential Smoothing (ETS) 带加法趋势和加法周季节性

4个预测渠道:
    - demand_total      : 总销量
    - demand_hospital   : 医院渠道销量
    - demand_chain      : 连锁药店渠道销量
    - demand_independent: 独立药店渠道销量

运行: python 03_model_ets.py
输出: results/ets_predictions.csv, results/ets_metrics.csv
"""

# =============================================================================
# 第一部分: 导入必要的库
# =============================================================================

import os                          # 操作系统接口，用于文件和目录操作
import warnings                    # 用于屏蔽不必要的警告信息
import numpy as np                 # 数值计算库，提供高效的数组运算
import pandas as pd                # 数据处理库，用于读取和操作表格数据
from datetime import datetime      # 日期时间处理
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # ETS指数平滑模型

# 屏蔽statsmodels在模型拟合过程中可能产生的各种警告，保持输出整洁
warnings.filterwarnings("ignore", module="statsmodels")


# =============================================================================
# 第二部分: 定义全局常量和配置参数
# =============================================================================

# 获取当前脚本所在目录，确保数据文件路径正确
# os.path.dirname(__file__) 获取脚本文件所在的目录路径
# os.getcwd() 获取当前工作目录（作为备选方案）
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

# 定义数据文件的路径，基于当前脚本所在目录的相对路径
# demand_daily.csv 包含每日的药品需求数据（date, sku_id, demand_total等列）
DEMAND_FILE = os.path.join(BASE_DIR, "..", "data", "demand_daily.csv")

# sku_profiles.csv 包含每个SKU的画像信息（sku_id, demand_class等列）
PROFILES_FILE = os.path.join(BASE_DIR, "..", "data", "sku_profiles.csv")

# 定义输出目录路径，基于当前脚本所在目录，用于存放预测结果和评估指标
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# 定义预测结果输出文件路径（每个SKU未来30天的4渠道预测值）
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "ets_predictions.csv")

# 定义评估指标输出文件路径（每个SKU每个渠道的MAPE和RMSE）
METRICS_FILE = os.path.join(RESULTS_DIR, "ets_metrics.csv")

# 定义训练集的时间范围：模型将使用此时间段内的数据进行训练
# 从2020-01-01到2024-12-31，约5年的日度数据
TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"

# 定义验证集的时间范围：用于评估模型预测精度
# 从2025-01-01到2026-05-31，约17个月的日度数据
VALID_START = "2025-01-01"
VALID_END = "2026-05-31"

# 定义预测目标时间段：预测未来90天的需求
# 从2026-06-01到2026-08-30
FORECAST_START = "2026-06-01"
FORECAST_END = "2026-08-30"

# 定义需要预测的4个渠道列名列表
# 这4列分别对应总销量、医院渠道、连锁药店渠道、独立药店渠道
CHANNEL_COLUMNS = [
    "demand_total",       # 总销量
    "demand_hospital",    # 医院渠道销量
    "demand_chain",       # 连锁药店渠道销量
    "demand_independent"  # 独立药店渠道销量
]

# 定义渠道显示名称映射，用于输出结果中的列名
# 将原始列名映射为更友好的输出列名
CHANNEL_OUTPUT_NAMES = {
    "demand_total": "forecast_total",       # 总销量预测列名
    "demand_hospital": "forecast_hospital", # 医院渠道预测列名
    "demand_chain": "forecast_chain",       # 连锁药店渠道预测列名
    "demand_independent": "forecast_independent"  # 独立药店渠道预测列名
}

# 定义渠道简称映射，用于评估指标输出
# 将原始列名映射为简短的渠道标识
CHANNEL_SHORT_NAMES = {
    "demand_total": "total",        # 总销量简称
    "demand_hospital": "hospital",  # 医院渠道简称
    "demand_chain": "chain",        # 连锁药店渠道简称
    "demand_independent": "independent"  # 独立药店渠道简称
}

# ETS模型配置参数（修正后，解决平线问题）：
# error参数: 'add' 表示加法误差（适用于需求数据波动相对稳定的情况）
# trend参数: 'add' 表示加法趋势项（捕捉需求的上升或下降趋势）
# seasonal参数: 'add' 表示加法季节性项（捕捉周期性波动）
# seasonal_periods: 7 表示周季节性（7天周期，捕捉周末效应）
# damped_trend: True 使用阻尼趋势，防止趋势无限增长
ETS_ERROR = "add"           # 误差类型：加法误差
ETS_TREND = "add"           # 趋势类型：加法趋势（必须有，避免平线）
ETS_SEASONAL = "add"        # 季节类型：加法季节（必须有，捕捉周期性）
ETS_SEASONAL_PERIODS = 7    # 季节性周期：7天（周季节性，捕捉周末效应）
ETS_DAMPED = True           # 阻尼趋势：True（防止趋势无限外推）


# =============================================================================
# 第三部分: 辅助函数定义
# =============================================================================

def ensure_dir(directory):
    """
    确保指定目录存在，如果不存在则创建它。
    
    参数:
        directory (str): 需要检查或创建的目录路径
    
    说明:
        使用 exist_ok=True 参数可以避免在目录已存在时抛出异常，
        这样脚本可以多次运行而不会报错。
    """
    # os.makedirs递归创建目录（如果父目录也不存在会自动创建）
    # exist_ok=True表示如果目录已存在则不报错
    os.makedirs(directory, exist_ok=True)


def compute_mape(actual, predicted):
    """
    计算MAPE（平均绝对百分比误差）。
    
    MAPE是衡量预测精度的常用指标，表示预测值与真实值之间
    的百分比差异的平均值。值越小表示预测越准确。
    
    公式: MAPE = mean(|(actual - predicted) / actual|) * 100
    
    参数:
        actual (np.array): 真实值数组（验证集的实际需求）
        predicted (np.array): 预测值数组（模型在验证期内的预测需求）
    
    返回:
        float: MAPE值，单位为百分比（如5.0表示5%）
               如果actual全为0，返回np.nan避免除零错误
    """
    # 将输入转换为numpy数组，确保可以进行数值运算
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    # 检查实际值中是否有非零值，避免除以零
    # 使用1e-12作为阈值判断数组是否全为零（考虑浮点数精度问题）
    if np.sum(np.abs(actual)) < 1e-12:
        # 如果所有实际值都为0，MAPE无意义，返回NaN
        return np.nan
    
    # 只计算实际值不为0的数据点的MAPE，避免除以零
    # mask是一个布尔数组，标记actual中非零的位置
    mask = actual != 0
    if mask.sum() == 0:
        # 如果所有实际值都为0（双重检查），返回NaN
        return np.nan
    
    # 计算每个数据点的绝对百分比误差：|(真实值 - 预测值) / 真实值|
    # 然后对所有数据点的误差取平均值
    # 最后乘以100将结果转换为百分比形式
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return mape


def compute_rmse(actual, predicted):
    """
    计算RMSE（均方根误差）。
    
    RMSE衡量预测值与真实值之间的偏差程度，对大误差更敏感。
    值越小表示预测越准确，单位与原始数据相同。
    
    公式: RMSE = sqrt(mean((actual - predicted)^2))
    
    参数:
        actual (np.array): 真实值数组
        predicted (np.array): 预测值数组
    
    返回:
        float: RMSE值，单位与需求数量相同
    """
    # 将输入转换为numpy数组，确保可以进行数值运算
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    # 计算每个数据点的预测误差：(真实值 - 预测值)
    # 对每个误差取平方：(真实值 - 预测值)^2
    # 对所有数据点的平方误差取平均值，得到均方误差(MSE)
    mse = np.mean((actual - predicted) ** 2)
    
    # 对均方误差取平方根，得到均方根误差(RMSE)
    # RMSE的单位与原始数据一致，便于理解
    rmse = np.sqrt(mse)
    return rmse


def prepare_time_series(sku_data, channel_col):
    """
    将SKU原始数据转换为规范的时间序列格式。
    
    该函数负责：
    1. 按日期排序
    2. 将数据转换为以日期为索引的pd.Series
    3. 填充缺失日期（用0填充，表示当天无销售记录）
    4. 确保时间序列连续性
    
    参数:
        sku_data (pd.DataFrame): 某个SKU的原始数据，包含date列和渠道列
        channel_col (str): 渠道列名（如'demand_total'）
    
    返回:
        pd.Series: 规范的时间序列，索引为完整日期范围，值为渠道销量
    """
    # 按日期升序排序，确保时间序列数据是按时间顺序排列的
    # 时间序列模型严格要求数据必须按时间顺序输入
    # reset_index(drop=True)重置索引，丢弃旧索引
    sku_data = sku_data.sort_values("date").reset_index(drop=True)
    
    # 将数据转换为时间序列格式：以date为索引，指定渠道列为值
    # set_index("date")将date列设为索引
    # [channel_col]选取指定渠道的需求列，结果是pd.Series
    # ETSModel需要pd.Series格式，索引为DatetimeIndex
    ts = sku_data.set_index("date")[channel_col]
    
    # 确保时间序列是连续的日度数据，没有缺失日期
    # pd.date_range生成从最小日期到最大日期的完整日期序列
    # freq="D"表示按天频率
    full_date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq="D")
    
    # reindex按照完整日期序列重新索引
    # 如果有缺失日期，用0填充（表示当天无销售记录）
    # 这是药品需求数据的常见情况：周末或节假日可能没有销售
    ts = ts.reindex(full_date_range, fill_value=0)
    
    # 返回规范后的时间序列
    return ts


def fit_ets_model(train_ts):
    """
    拟合ETS模型并返回拟合结果。
    
    使用加法趋势 + 加法周季节性(周期7天)的配置，
    能够捕捉需求的趋势变化和周末效应，避免预测出平线。
    
    参数:
        train_ts (pd.Series): 训练数据时间序列，索引为DatetimeIndex
    
    返回:
        ETSResults: 拟合好的ETS模型结果对象
        None: 如果拟合失败则返回None
    """
    try:
        # 创建ETSModel实例
        # ETSModel是statsmodels库中的指数平滑状态空间模型
        # 参数说明：
        #   train_ts: 训练数据（pd.Series，索引必须是DatetimeIndex）
        #   error='add': 加法误差，适合波动相对稳定的数据
        #   trend='add': 加法趋势，捕捉需求的上升或下降趋势（关键！避免平线）
        #   seasonal='add': 加法季节性，捕捉周期性波动（关键！避免平线）
        #   seasonal_periods=7: 周季节性，7天周期捕捉周末效应
        #   damped_trend=True: 阻尼趋势，防止趋势无限外推
        ets_model = ETSModel(
            train_ts,                         # 训练数据（pd.Series格式）
            error=ETS_ERROR,                  # 误差类型：加法
            trend=ETS_TREND,                  # 趋势类型：加法（必须有趋势）
            seasonal=ETS_SEASONAL,            # 季节类型：加法（必须有季节性）
            seasonal_periods=ETS_SEASONAL_PERIODS,  # 季节性周期：7天（周季节性）
            damped_trend=ETS_DAMPED           # 阻尼趋势：True
        )
        
        # 拟合模型：使用最大似然估计(MLE)方法估计模型参数
        # disp=False表示不显示优化过程中的迭代信息，保持输出整洁
        ets_result = ets_model.fit(disp=False)
        
        # 返回拟合成功的模型结果
        return ets_result
        
    except Exception as e:
        # 如果模型拟合失败（如数据异常、参数不收敛等），
        # 捕获异常并返回None，让上层函数处理
        return None


def forecast_channel(ets_result, train_end, forecast_start, forecast_end):
    """
    使用拟合好的ETS模型进行预测。
    
    参数:
        ets_result (ETSResults): 拟合好的ETS模型结果
        train_end (str): 训练集结束日期
        forecast_start (str): 预测开始日期
        forecast_end (str): 预测结束日期
    
    返回:
        pd.Series: 目标预测时间段内的预测值序列
    """
    # 计算从训练集结束次日到预测目标期结束的总天数
    # 这决定了模型需要向前预测的总步数
    steps_to_forecast_end = (pd.Timestamp(forecast_end) - pd.Timestamp(train_end)).days
    
    # 使用拟合好的模型进行预测
    # forecast方法返回从训练集结束次日开始的预测值序列
    forecast_result = ets_result.forecast(steps=steps_to_forecast_end)
    
    # 从预测结果中截取目标预测时间段
    # 只保留预测目标月份的数据
    forecast_result = forecast_result[
        (forecast_result.index >= forecast_start) & 
        (forecast_result.index <= forecast_end)
    ]
    
    # 返回截取后的预测结果
    return forecast_result


def evaluate_channel(ets_result, ts, valid_start, valid_end, train_end):
    """
    在验证集上评估ETS模型的预测性能。
    
    参数:
        ets_result (ETSResults): 拟合好的ETS模型结果
        ts (pd.Series): 完整时间序列数据
        valid_start (str): 验证集开始日期
        valid_end (str): 验证集结束日期
        train_end (str): 训练集结束日期
    
    返回:
        tuple: (mape, rmse) 两个评估指标
               如果无法计算则返回 (np.nan, np.nan)
    """
    # 从完整时间序列中截取验证集数据
    # 验证集用于评估模型的泛化能力和预测精度
    valid_ts = ts[(ts.index >= valid_start) & (ts.index <= valid_end)]
    
    # 初始化评估指标变量为NaN
    # 如果后续无法计算（如验证集为空），则保持NaN
    mape_val = np.nan
    rmse_val = np.nan
    
    # 检查验证集是否有足够的数据进行评估
    if len(valid_ts) > 0:
        # 计算从训练集结束次日到验证集结束的总天数
        # 这决定了模型需要向前预测多少步才能覆盖验证集
        steps_to_valid_end = (pd.Timestamp(valid_end) - pd.Timestamp(train_end)).days
        
        # 使用拟合好的模型进行预测
        # forecast方法返回从训练集结束次日开始的预测值序列
        forecast_valid = ets_result.forecast(steps=steps_to_valid_end)
        
        # 从预测结果中截取验证集对应的时间段
        # 因为forecast返回的预测范围可能比验证集更大
        forecast_valid = forecast_valid[
            (forecast_valid.index >= valid_start) & 
            (forecast_valid.index <= valid_end)
        ]
        
        # 确保验证集和预测集有相同的日期索引，才能进行逐日比较
        # intersection取两个索引的交集
        common_dates = valid_ts.index.intersection(forecast_valid.index)
        
        # 如果存在共同的日期，则计算评估指标
        if len(common_dates) > 0:
            # 获取共同日期对应的真实值（从验证集中提取）
            actual_vals = valid_ts.loc[common_dates].values
            # 获取共同日期对应的预测值（从预测结果中提取）
            pred_vals = forecast_valid.loc[common_dates].values
            
            # 调用辅助函数计算MAPE和RMSE
            mape_val = compute_mape(actual_vals, pred_vals)
            rmse_val = compute_rmse(actual_vals, pred_vals)
    
    # 返回评估指标元组
    return mape_val, rmse_val


# =============================================================================
# 第四部分: 主程序
# =============================================================================

def main():
    """
    主函数：执行ETS模型训练和预测的完整流程。
    
    流程步骤:
        1. 读取需求数据和SKU画像
        2. 筛选fast类型的SKU
        3. 对每个fast SKU、每个渠道分别进行数据预处理
        4. 分别训练4个ETS模型（每个渠道一个）
        5. 在验证集上分别评估每个渠道模型的性能
        6. 分别预测未来90天每个渠道的需求
        7. 保存预测结果和评估指标
    """
    
    # 在控制台打印程序开始运行的信息，方便用户了解进度
    # "=" * 70 生成70个等号作为分隔线
    print("=" * 70)
    print("ETS指数平滑模型 — Fast SKU 4渠道需求预测")
    print("模型配置: 加法趋势 + 加法周季节性(周期7天)")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 步骤1: 确保输出目录存在
    # -------------------------------------------------------------------------
    # 调用ensure_dir函数创建results目录（如果不存在的话）
    # 这是为了确保后续文件保存操作不会因为目录不存在而失败
    ensure_dir(RESULTS_DIR)
    print(f"\n[1] 输出目录已就绪: {RESULTS_DIR}/")
    
    # -------------------------------------------------------------------------
    # 步骤2: 读取需求数据
    # -------------------------------------------------------------------------
    # 使用pandas的read_csv函数读取日度需求数据
    # 这是一个大型CSV文件，包含500个SKU从2020到2026年的日度需求
    print(f"\n[2] 正在读取需求数据: {DEMAND_FILE} ...")
    
    # 读取所有需要的列：日期、SKU编号、以及4个渠道的销量列
    # 这样4个渠道的数据都可以用于建模
    demand_df = pd.read_csv(DEMAND_FILE)
    
    # 将date列从字符串类型转换为pandas的datetime类型
    # 这是后续按日期进行筛选和分组的前提条件
    # to_datetime函数会自动识别各种日期格式
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    
    # 打印数据基本信息，帮助确认数据读取正确
    # {:,} 格式化输出千分位分隔符（如 1,171,500）
    print(f"     需求数据行数: {len(demand_df):,}")
    print(f"     日期范围: {demand_df['date'].min().date()} ~ {demand_df['date'].max().date()}")
    print(f"     SKU数量: {demand_df['sku_id'].nunique()}")
    print(f"     渠道列: {CHANNEL_COLUMNS}")
    
    # -------------------------------------------------------------------------
    # 步骤3: 读取SKU画像数据
    # -------------------------------------------------------------------------
    # sku_profiles.csv包含每个SKU的分类标签（fast/seasonal/long_tail/policy_shocked）
    print(f"\n[3] 正在读取SKU画像: {PROFILES_FILE} ...")
    
    # 使用pandas读取SKU画像CSV文件
    profiles_df = pd.read_csv(PROFILES_FILE)
    
    # 从画像数据中筛选出demand_class为'fast'的SKU
    # fast类型的SKU特征是：销量高且稳定，波动小，适合用ETS模型
    # 使用布尔索引筛选：profiles_df["demand_class"] == "fast" 返回布尔数组
    fast_profiles = profiles_df[profiles_df["demand_class"] == "fast"]
    
    # 获取fast SKU的编号列表，用于后续筛选需求数据
    # unique()去重，tolist()转换为Python列表
    fast_sku_list = fast_profiles["sku_id"].unique().tolist()
    
    # 打印fast SKU的数量和前5个SKU编号
    print(f"     Fast SKU数量: {len(fast_sku_list)}")
    print(f"     Fast SKU示例: {fast_sku_list[:5]} ...")
    
    # -------------------------------------------------------------------------
    # 步骤4: 筛选fast SKU的需求数据
    # -------------------------------------------------------------------------
    # 使用isin方法从全部需求数据中筛选出fast SKU的数据
    # isin(fast_sku_list) 返回布尔数组，标记哪些行属于fast SKU
    # 这样可以大幅减少数据量，只保留需要建模的SKU
    fast_demand = demand_df[demand_df["sku_id"].isin(fast_sku_list)].copy()
    
    # 打印筛选后的数据信息
    print(f"\n[4] Fast SKU需求数据行数: {len(fast_demand):,}")
    
    # -------------------------------------------------------------------------
    # 步骤5: 初始化结果存储列表
    # -------------------------------------------------------------------------
    # predictions_list用于存储所有fast SKU的未来90天4渠道预测结果
    # 每个元素是一个字典，包含sku_id、date、以及4个渠道的预测值
    predictions_list = []
    
    # metrics_list用于存储所有fast SKU每个渠道的评估指标（MAPE和RMSE）
    # 每个元素是一个字典，包含sku_id、channel、mape、rmse
    metrics_list = []
    
    # success_count计数器：记录成功训练和预测的SKU+渠道组合数量
    success_count = 0
    
    # failed_count计数器：记录失败（出现错误）的SKU+渠道组合数量
    failed_count = 0
    
    # skipped_count计数器：记录因全零数据而跳过的渠道数量
    skipped_count = 0
    
    # -------------------------------------------------------------------------
    # 步骤6: 对每个fast SKU分别训练ETS模型（每个SKU训练4个模型，每个渠道一个）
    # -------------------------------------------------------------------------
    # groupby("sku_id")按照SKU编号分组，每个SKU独立处理
    # 因为不同fast SKU的需求模式各不相同，需要独立建模
    grouped = fast_demand.groupby("sku_id")
    
    # 获取需要处理的SKU总数，用于显示进度
    total_skus = len(grouped)
    
    # 如果没有fast SKU，打印警告信息并提前退出
    if total_skus == 0:
        print("\n[警告] 没有找到任何fast类型的SKU，程序退出。")
        return
    
    print(f"\n[5] 开始对 {total_skus} 个Fast SKU进行ETS建模...")
    print(f"     每个SKU训练 {len(CHANNEL_COLUMNS)} 个模型（每个渠道一个）")
    print("-" * 70)
    
    # 遍历每个fast SKU的数据
    # enumerate从1开始计数，用于显示当前处理第几个SKU
    for idx, (sku_id, sku_data) in enumerate(grouped, start=1):
        # 打印当前处理的SKU编号和进度（如 [1/39] SKU-0014）
        print(f"\n[{idx}/{total_skus}] 处理 SKU: {sku_id}")
        
        # 初始化一个字典，用于存储该SKU所有渠道的预测结果
        # 键为日期，值为包含4个渠道预测值的字典
        sku_forecasts_by_date = {}
        
        # 标记该SKU是否有至少一个渠道成功建模
        sku_has_success = False
        
        # 对该SKU的每个渠道分别训练模型
        # CHANNEL_COLUMNS包含4个渠道列名
        for channel_idx, channel_col in enumerate(CHANNEL_COLUMNS):
            # 获取该渠道的简称，用于输出显示
            channel_short = CHANNEL_SHORT_NAMES[channel_col]
            
            try:
                # ---- 6.1 数据预处理 ----
                
                # 调用prepare_time_series函数将原始数据转换为规范的时间序列
                # 该函数内部完成：排序、设置索引、填充缺失日期
                ts = prepare_time_series(sku_data, channel_col)
                
                # ---- 6.2 划分训练集 ----
                
                # 从完整时间序列中截取训练集数据
                # 训练集用于拟合ETS模型的参数
                # 使用布尔索引筛选在训练集时间范围内的数据
                train_ts = ts[(ts.index >= TRAIN_START) & (ts.index <= TRAIN_END)]
                
                # 检查训练集是否有足够的数据点
                # ETS模型（含趋势和季节性）需要足够的数据点
                # seasonal_periods=7，至少需要2个完整周期（14天）再加一些数据
                # 这里保守地要求至少30天数据
                if len(train_ts) < 30:
                    # 如果训练数据不足，打印警告信息并跳过此渠道
                    print(f"     [{channel_short}] 训练数据不足 ({len(train_ts)} 天)，跳过")
                    failed_count += 1  # 增加失败计数
                    continue           # 跳过后续代码，处理下一个渠道
                
                # 确保训练数据中没有NaN值，NaN会导致模型拟合失败
                # fillna(0)用0填充NaN（假设缺失值表示当天无销售）
                train_ts = train_ts.fillna(0)
                
                # 检查该渠道的训练数据是否全为0
                # 如果全为0，则无需建模，直接预测0即可
                if train_ts.sum() < 1e-12:
                    # 打印信息说明该渠道全为0
                    print(f"     [{channel_short}] 训练数据全为0，预测值填0")
                    skipped_count += 1  # 增加跳过计数
                    
                    # 为验证集评估：MAPE填NaN（全为0无法计算MAPE），RMSE为0
                    # 生成全0的预测结果
                    mape_val = np.nan
                    rmse_val = 0.0
                    
                    # 生成未来90天的全0预测
                    # pd.date_range生成预测时间段的完整日期序列
                    future_dates = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq="D")
                    # 创建全0的预测序列，索引为预测日期
                    forecast_future = pd.Series(0.0, index=future_dates)
                    
                    # 将全0预测结果保存到sku_forecasts_by_date字典中
                    # 遍历每个预测日期和预测值（这里都是0）
                    for forecast_date, forecast_value in forecast_future.items():
                        # 如果该日期还没有在字典中，创建新的字典项
                        if forecast_date not in sku_forecasts_by_date:
                            sku_forecasts_by_date[forecast_date] = {}
                        # 将当前渠道的预测值填入字典
                        # 使用输出列名作为键
                        sku_forecasts_by_date[forecast_date][CHANNEL_OUTPUT_NAMES[channel_col]] = 0.0
                    
                    # 保存评估指标（MAPE为NaN，RMSE为0）
                    metrics_list.append({
                        "sku_id": sku_id,                    # SKU编号
                        "channel": channel_short,            # 渠道简称
                        "mape": mape_val,                    # MAPE填NaN（全为0无法计算）
                        "rmse": rmse_val                     # RMSE为0（预测值和真实值都是0）
                    })
                    
                    # 标记该SKU有成功处理的渠道
                    sku_has_success = True
                    # 跳过ETS建模，继续处理下一个渠道
                    continue
                
                # ---- 6.3 训练ETS模型 ----
                
                # 调用fit_ets_model函数拟合ETS模型
                # 使用加法趋势 + 加法周季节性配置，避免平线问题
                ets_result = fit_ets_model(train_ts)
                
                # 检查模型拟合是否成功
                if ets_result is None:
                    # 如果拟合失败，打印警告并跳过此渠道
                    print(f"     [{channel_short}] ETS模型拟合失败，跳过")
                    failed_count += 1  # 增加失败计数
                    continue           # 跳过后续代码，处理下一个渠道
                
                # 打印模型拟合成功的信息，包括AIC（赤池信息准则）
                # AIC是衡量模型拟合优度的指标，越小越好
                print(f"     [{channel_short}] ETS模型拟合成功 (AIC: {ets_result.aic:.2f})")
                
                # ---- 6.4 在验证集上评估模型 ----
                
                # 调用evaluate_channel函数在验证集上评估模型性能
                # 传入拟合好的模型、完整时间序列、验证集起止日期、训练集结束日期
                mape_val, rmse_val = evaluate_channel(
                    ets_result, ts, VALID_START, VALID_END, TRAIN_END
                )
                
                # 打印评估结果，保留两位小数
                print(f"          验证集MAPE: {mape_val:.2f}% | RMSE: {rmse_val:.2f}")
                
                # ---- 6.5 预测未来90天 ----
                
                # 调用forecast_channel函数预测未来30天的需求
                # 传入拟合好的模型、训练集结束日期、预测起止日期
                forecast_future = forecast_channel(
                    ets_result, TRAIN_END, FORECAST_START, FORECAST_END
                )
                
                # 打印预测结果的天数
                print(f"          未来90天预测: {len(forecast_future)} 天")
                
                # ---- 6.6 保存该SKU+渠道的预测结果 ----
                
                # 将预测结果保存到sku_forecasts_by_date字典中
                # 遍历forecast_future序列的每个元素（日期和预测值）
                for forecast_date, forecast_value in forecast_future.items():
                    # 如果该日期还没有在字典中，创建新的字典项
                    if forecast_date not in sku_forecasts_by_date:
                        sku_forecasts_by_date[forecast_date] = {}
                    # 将当前渠道的预测值填入字典
                    # 使用输出列名作为键（如 forecast_total）
                    # 确保预测值不为负数（需求不能为负）
                    sku_forecasts_by_date[forecast_date][CHANNEL_OUTPUT_NAMES[channel_col]] = max(0.0, forecast_value)
                
                # ---- 6.7 保存该SKU+渠道的评估指标 ----
                
                # 将MAPE和RMSE添加到metrics_list列表中
                metrics_list.append({
                    "sku_id": sku_id,          # SKU编号
                    "channel": channel_short,  # 渠道简称
                    "mape": mape_val,          # 平均绝对百分比误差（%）
                    "rmse": rmse_val           # 均方根误差
                })
                
                # 增加成功计数
                success_count += 1
                
                # 标记该SKU有成功处理的渠道
                sku_has_success = True
                
            except Exception as e:
                # 如果建模过程中出现任何错误（如数据异常、模型不收敛等），
                # 捕获异常并打印错误信息，然后跳过此SKU+渠道组合继续处理下一个
                # 这种设计保证单个SKU+渠道组合的失败不会影响其他组合的处理
                print(f"     [{channel_short}] 错误: {str(e)}")
                failed_count += 1
                continue
        
        # ---- 6.8 将该SKU所有渠道的预测结果合并保存 ----
        
        # 检查该SKU是否有成功处理的渠道
        if sku_has_success and len(sku_forecasts_by_date) > 0:
            # 遍历该SKU所有预测日期的结果
            for forecast_date, channel_values in sku_forecasts_by_date.items():
                # 创建一个字典存储该SKU+日期的完整预测结果
                row = {
                    "sku_id": sku_id,       # SKU编号
                    "date": forecast_date    # 预测日期
                }
                # 将该日期所有4个渠道的预测值添加到字典中
                # 如果某个渠道没有预测值（建模失败），则填NaN
                for channel_col in CHANNEL_COLUMNS:
                    output_name = CHANNEL_OUTPUT_NAMES[channel_col]
                    # 从channel_values字典中获取预测值，如果不存在则填NaN
                    row[output_name] = channel_values.get(output_name, np.nan)
                
                # 将该行添加到predictions_list列表中
                predictions_list.append(row)
    
    # =============================================================================
    # 第五部分: 保存结果到CSV文件
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    # ---- 5.1 保存预测结果 ----
    
    # 检查是否有成功生成的预测结果
    if len(predictions_list) > 0:
        # 将预测结果列表转换为DataFrame
        # pd.DataFrame自动将字典列表转换为表格格式
        predictions_df = pd.DataFrame(predictions_list)
        
        # 确保日期列格式统一为字符串格式（YYYY-MM-DD）
        # 便于后续读取和处理，避免日期格式不一致的问题
        predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m-%d")
        
        # 确保输出列的顺序符合要求：sku_id, date, forecast_total, forecast_hospital, forecast_chain, forecast_independent
        # 按照指定顺序重新排列列
        output_columns = ["sku_id", "date"] + [CHANNEL_OUTPUT_NAMES[ch] for ch in CHANNEL_COLUMNS]
        # 只保留存在的列（防止某些渠道全部缺失）
        output_columns = [col for col in output_columns if col in predictions_df.columns]
        predictions_df = predictions_df[output_columns]
        
        # 将预测结果保存为CSV文件
        # index=False表示不保存行索引（只保存数据列）
        predictions_df.to_csv(PREDICTIONS_FILE, index=False)
        
        # 打印保存成功的信息
        print(f"\n[6] 预测结果已保存: {PREDICTIONS_FILE}")
        print(f"     总行数: {len(predictions_df):,}")
        print(f"     SKU数: {predictions_df['sku_id'].nunique()}")
        print(f"     日期范围: {predictions_df['date'].min()} ~ {predictions_df['date'].max()}")
        
        # 打印每个渠道的预测值统计摘要
        print(f"\n     各渠道预测值统计:")
        for channel_col in CHANNEL_COLUMNS:
            output_name = CHANNEL_OUTPUT_NAMES[channel_col]
            if output_name in predictions_df.columns:
                # 计算该渠道预测值的基本统计信息
                col_data = predictions_df[output_name]
                print(f"       {output_name}: 均值={col_data.mean():.2f}, 最小={col_data.min():.2f}, 最大={col_data.max():.2f}")
    else:
        # 如果没有任何成功的预测结果，打印警告
        print(f"\n[6] 警告: 没有成功的预测结果可保存")
    
    # ---- 5.2 保存评估指标 ----
    
    # 检查是否有成功计算的评估指标
    if len(metrics_list) > 0:
        # 将评估指标列表转换为DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        
        # 确保输出列的顺序：sku_id, channel, mape, rmse
        metrics_df = metrics_df[["sku_id", "channel", "mape", "rmse"]]
        
        # 将评估指标保存为CSV文件
        metrics_df.to_csv(METRICS_FILE, index=False)
        
        # 打印保存成功的信息和指标汇总统计
        print(f"\n[7] 评估指标已保存: {METRICS_FILE}")
        
        # 按渠道分组计算平均MAPE和RMSE
        print(f"\n     各渠道平均评估指标:")
        for channel_short in CHANNEL_SHORT_NAMES.values():
            # 筛选当前渠道的评估指标
            channel_metrics = metrics_df[metrics_df["channel"] == channel_short]
            if len(channel_metrics) > 0:
                # 计算该渠道的平均MAPE（排除NaN值）
                mean_mape = channel_metrics["mape"].mean()
                # 计算该渠道的平均RMSE（排除NaN值）
                mean_rmse = channel_metrics["rmse"].mean()
                # 打印该渠道的指标汇总
                print(f"       {channel_short}: 平均MAPE={mean_mape:.2f}%, 平均RMSE={mean_rmse:.2f}, SKU数={len(channel_metrics)}")
        
        # 计算并打印所有渠道的总体平均MAPE（排除NaN值）
        overall_mean_mape = metrics_df["mape"].mean()
        print(f"\n     总体平均MAPE: {overall_mean_mape:.2f}%")
        
        # 计算并打印所有渠道的总体平均RMSE（排除NaN值）
        overall_mean_rmse = metrics_df["rmse"].mean()
        print(f"     总体平均RMSE: {overall_mean_rmse:.2f}")
    else:
        # 如果没有任何评估指标，打印警告
        print(f"\n[7] 警告: 没有评估指标可保存")
    
    # ---- 5.3 打印最终处理统计 ----
    
    # 打印成功、失败和跳过的SKU+渠道组合数量统计
    total_combinations = total_skus * len(CHANNEL_COLUMNS)
    print(f"\n[8] 处理统计:")
    print(f"     SKU+渠道组合总数: {total_combinations}")
    print(f"     成功建模: {success_count}")
    print(f"     全零跳过: {skipped_count}")
    print(f"     失败: {failed_count}")
    
    # 打印程序正常结束的信息
    print("\n" + "=" * 70)
    print("ETS模型训练和预测完成！")
    print("=" * 70)


# =============================================================================
# 第六部分: 程序入口
# =============================================================================

if __name__ == "__main__":
    """
    程序入口点。
    
    当直接运行此脚本时（python 03_model_ets.py），会调用main函数。
    当作为模块被导入时（import），不会自动执行main函数。
    
    这是Python的标准做法，确保脚本既可以独立运行，也可以作为模块复用。
    """
    main()
