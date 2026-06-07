"""
药品需求预测数据生成器 — 01_generate_data.py
====================================================
为 Case 2: Pharmaceutical Demand Forecasting 生成完整的合成数据。

生成8个CSV文件:
1. products.csv      — 500个SKU主数据（含ATC-1~4拆解字段）
2. demand_daily.csv  — 日度需求（总体/医院/连锁/独立 4渠道）
3. external_signals.csv — 外部信号（CDC ILI%、百度指数、假期、VBP）
4. holidays.csv      — 中国假期日历（2020-2026）
5. sku_profiles.csv  — SKU需求画像分类
6. vbp_impact.csv    — 集采冲击参数
7. inventory.csv     — 日度库存快照
8. replenishment.csv — 补货建议基线参数

时间范围: 2020-01-01 至 2026-05-31（日度，~2333天）
SKU数量: 500个
渠道: 总体(total)、医院(hospital)、连锁(chain)、独立(independent)

运行: python 01_generate_data.py
输出: 保存在 ./data/ 目录下
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# 第一部分：全局配置
# ============================================================

# 设置随机种子，确保每次运行生成相同的数据（可复现）
SEED = 42
np.random.seed(SEED)

# 核心参数
N_SKUS = 500                    # SKU总数
START_DATE = "2020-01-01"       # 数据起始日期
END_DATE = "2026-05-31"         # 数据结束日期
HISTORY_END = pd.Timestamp("2026-05-31")  # 历史数据截止

# 集采（VBP）关键日期
VBP_DATE = pd.Timestamp("2021-06-01")     # 第一批集采扩围全国执行日期
VBP_DATE_2 = pd.Timestamp("2023-11-01")   # 第九批集采
VBP_PRICE_DROP = 0.51                      # 集采平均降价51%

# 治疗类别（therapy areas）— 对应老师的5-6个类别
THERAPY_AREAS = [
    "Cardiovascular",    # 心血管（C）— 包含氨氯地平
    "Antibiotics",       # 抗生素（J）
    "Diabetes",          # 糖尿病（A）
    "Respiratory",       # 呼吸系统（R）
    "CNS",               # 中枢神经系统（N）
    "Gastrointestinal",  # 消化系统（A）
]

# 创建输出目录
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("药品需求预测数据生成器")
print("=" * 60)
print(f"SKU数量: {N_SKUS}")
print(f"时间范围: {START_DATE} ~ {END_DATE}")
print(f"集采日期: {VBP_DATE.date()}")
print("=" * 60)


# ============================================================
# 第二部分：生成日期索引和外部信号
# ============================================================

print("\n[1/8] 生成日期索引和外部信号...")

# 生成完整的日期范围（日度）
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
n_days = len(date_range)
print(f"    总天数: {n_days}天")

# --- 外部信号表 external_signals.csv ---

# 2.1 流感活动指数（模拟CDC ILI%）
# 逻辑：流感季（11月-3月）指数高，其他时间低，有年际波动

def generate_flu_index(dates):
    """
    生成流感活动指数（0-10）
    模拟中国疾控中心ILI%数据的模式：
    - 每年11月-3月为流感季，指数较高
    - 1-2月通常达到峰值
    - 不同年份流行强度不同（2020低因为防疫，2023高因为放开）
    """
    flu_index = np.zeros(len(dates))
    
    for i, date in enumerate(dates):
        month = date.month
        year = date.year
        
        # 基础季节性：11月-3月是流感季
        if month in [11, 12, 1, 2, 3]:
            # 1-2月最厉害
            if month in [1, 2]:
                base = 6.0 + np.random.normal(0, 1.5)
            elif month == 12:
                base = 4.0 + np.random.normal(0, 1.0)
            else:  # 11月, 3月
                base = 3.0 + np.random.normal(0, 0.8)
        else:
            base = 0.5 + np.random.normal(0, 0.3)
        
        # 2020年：疫情初期严格防疫 → 流感极低
        if year == 2020:
            base *= 0.3
        # 2021年：防疫常态 → 流感略低
        elif year == 2021:
            base *= 0.6
        # 2022年底-2023年初：放开后流感反弹厉害
        elif year == 2023 and month <= 3:
            base *= 1.8
        # 2023年：流感活动恢复
        elif year == 2023:
            base *= 1.2
        # 2024-2026：正常年际波动
        elif year >= 2024:
            year_factor = 0.8 + np.random.random() * 0.4  # 0.8-1.2随机
            base *= year_factor
        
        flu_index[i] = max(0, min(10, base))  # 裁剪到0-10
    
    return flu_index


# 2.2 百度流感搜索指数
# 逻辑：与流感活动正相关，但有一定领先性（人们先搜再去医院）

def generate_baidu_index(dates, flu_index):
    """
    生成百度流感搜索指数（0-10000）
    与flu_index正相关，但振幅更大，有一些独立波动
    """
    baidu_index = flu_index * 800 + np.random.normal(500, 200, len(dates))
    # 偶尔有独立搜索高峰（如新闻事件驱动的恐慌性搜索）
    for i, date in enumerate(dates):
        if np.random.random() < 0.02:  # 2%概率出现搜索小高峰
            baidu_index[i] += np.random.uniform(1000, 3000)
    return np.clip(baidu_index, 100, 10000)


# 2.3 假期标记

def generate_holiday_flags(dates):
    """
    标记中国法定节假日
    返回: is_holiday(是否假日), is_workday(是否调休工作日)
    """
    # 中国主要假期定义（简化版）
    # 格式: (月, 日, 假期名称)
    holidays_2020_2026 = {
        # 春节（农历，这里用近似日期）
        (1, 24, 2020): "Spring Festival", (2, 11, 2021): "Spring Festival",
        (1, 31, 2022): "Spring Festival", (1, 21, 2023): "Spring Festival",
        (2, 9, 2024): "Spring Festival", (1, 28, 2025): "Spring Festival",
        (2, 16, 2026): "Spring Festival",
        # 清明
        (4, 4, 2020): "Qingming", (4, 3, 2021): "Qingming",
        (4, 5, 2022): "Qingming", (4, 5, 2023): "Qingming",
        (4, 4, 2024): "Qingming", (4, 4, 2025): "Qingming",
        (4, 5, 2026): "Qingming",
        # 五一
        (5, 1, 2020): "Labor Day", (5, 1, 2021): "Labor Day",
        (5, 1, 2022): "Labor Day", (5, 1, 2023): "Labor Day",
        (5, 1, 2024): "Labor Day", (5, 1, 2025): "Labor Day",
        (5, 1, 2026): "Labor Day",
        # 端午（农历，近似）
        (6, 25, 2020): "Dragon Boat", (6, 14, 2021): "Dragon Boat",
        (6, 3, 2022): "Dragon Boat", (6, 22, 2023): "Dragon Boat",
        (6, 10, 2024): "Dragon Boat", (5, 31, 2025): "Dragon Boat",
        (6, 18, 2026): "Dragon Boat",
        # 国庆
        (10, 1, 2020): "National Day", (10, 1, 2021): "National Day",
        (10, 1, 2022): "National Day", (10, 1, 2023): "National Day",
        (10, 1, 2024): "National Day", (10, 1, 2025): "National Day",
        (10, 1, 2026): "National Day",
    }
    
    is_holiday = np.zeros(len(dates), dtype=int)
    holiday_names = [""] * len(dates)
    
    for i, date in enumerate(dates):
        key = (date.month, date.day, date.year)
        if key in holidays_2020_2026:
            is_holiday[i] = 1
            holiday_names[i] = holidays_2020_2026[key]
    
    return is_holiday, holiday_names


# 生成外部信号
flu_index = generate_flu_index(date_range)
baidu_index = generate_baidu_index(date_range, flu_index)
is_holiday, holiday_names = generate_holiday_flags(date_range)

# 计算距离最近节假日的天数
days_to_holiday = np.zeros(len(date_range), dtype=int)
for i in range(len(date_range)):
    # 向前和向后查找最近的假期
    min_dist = 365
    for j in range(len(date_range)):
        if is_holiday[j] and abs(i - j) < min_dist:
            min_dist = abs(i - j)
    days_to_holiday[i] = min_dist

# 构建external_signals.csv
external_signals = pd.DataFrame({
    'date': date_range,
    'year': date_range.year,
    'month': date_range.month,
    'day': date_range.day,
    'day_of_week': date_range.dayofweek,  # 0=周一
    'week_of_year': date_range.isocalendar().week.values,
    'quarter': date_range.quarter,
    'is_weekend': (date_range.dayofweek >= 5).astype(int),
    'is_holiday': is_holiday,
    'days_to_nearest_holiday': days_to_holiday,
    'season': date_range.month.map({3:'spring',4:'spring',5:'spring',
                                     6:'summer',7:'summer',8:'summer',
                                     9:'autumn',10:'autumn',11:'autumn',
                                     12:'winter',1:'winter',2:'winter'}),
    'is_flu_season': ((date_range.month <= 3) | (date_range.month >= 11)).astype(int),
    'flu_activity_index': np.round(flu_index, 2),
    'is_allergy_season': date_range.month.isin([3, 4, 5]).astype(int),
    'baidu_flu_index': np.round(baidu_index, 0).astype(int),
    'is_vbp_period': ((date_range >= VBP_DATE) | (date_range >= VBP_DATE_2)).astype(int),
    'days_since_vbp1': np.clip((date_range - VBP_DATE).days, -1, None),
    'days_since_vbp2': np.clip((date_range - VBP_DATE_2).days, -1, None),
})

external_signals.to_csv("data/external_signals.csv", index=False)
print(f"    [OK] external_signals.csv: {len(external_signals)}行 × {len(external_signals.columns)}列")

# 生成holidays.csv（假期日历）
holidays_df = external_signals[external_signals['is_holiday'] == 1][['date', 'year', 'month', 'day']].copy()
holidays_df['holiday_name'] = [holiday_names[i] for i in holidays_df.index]
holidays_df.to_csv("data/holidays.csv", index=False)
print(f"    [OK] holidays.csv: {len(holidays_df)}个假期")


# ============================================================
# 第三部分：生成500个SKU的产品主数据
# ============================================================

print(f"\n[2/8] 生成 {N_SKUS} 个SKU的产品主数据...")

# 药品通用名池（按治疗领域分类）
DRUG_POOL = {
    "Cardiovascular": [
        ("Amlodipine", "氨氯地平", "C08CA01"),
        ("Metoprolol", "美托洛尔", "C07AB02"),
        ("Valsartan", "缬沙坦", "C09CA03"),
        ("Atorvastatin", "阿托伐他汀", "C10AA05"),
        ("Simvastatin", "辛伐他汀", "C10AA01"),
        ("Losartan", "氯沙坦", "C09CA01"),
        ("Bisoprolol", "比索洛尔", "C07AB07"),
        ("Rosuvastatin", "瑞舒伐他汀", "C10AA07"),
    ],
    "Antibiotics": [
        ("Amoxicillin", "阿莫西林", "J01CA04"),
        ("Azithromycin", "阿奇霉素", "J01FA10"),
        ("Cefuroxime", "头孢呋辛", "J01DC02"),
        ("Levofloxacin", "左氧氟沙星", "J01MA12"),
        ("Metronidazole", "甲硝唑", "J01XD01"),
        ("Cefixime", "头孢克肟", "J01DD08"),
        ("Clarithromycin", "克拉霉素", "J01FA09"),
        ("Moxifloxacin", "莫西沙星", "J01MA14"),
    ],
    "Diabetes": [
        ("Metformin", "二甲双胍", "A10BA02"),
        ("Glimepiride", "格列美脲", "A10BB12"),
        ("Sitagliptin", "西格列汀", "A10BH01"),
        ("Empagliflozin", "恩格列净", "A10BK03"),
        ("Gliclazide", "格列齐特", "A10BB09"),
        ("Linagliptin", "利格列汀", "A10BH05"),
        ("Acarbose", "阿卡波糖", "A10BF01"),
        ("Repaglinide", "瑞格列奈", "A10BX02"),
    ],
    "Respiratory": [
        ("Oseltamivir", "奥司他韦", "J05AH02"),
        ("Montelukast", "孟鲁司特", "R03DC03"),
        ("Salbutamol", "沙丁胺醇", "R03AC02"),
        ("Ambroxol", "氨溴索", "R05CB06"),
        ("Budesonide", "布地奈德", "R03BA02"),
        ("Dextromethorphan", "右美沙芬", "R05DA09"),
        ("Carbocysteine", "羧甲司坦", "R05CB03"),
        ("Ipratropium", "异丙托溴铵", "R03BB01"),
    ],
    "CNS": [
        ("Donepezil", "多奈哌齐", "N06DA02"),
        ("Escitalopram", "艾司西酞普兰", "N06AB10"),
        ("Gabapentin", "加巴喷丁", "N03AX12"),
        ("Pregabalin", "普瑞巴林", "N03AX16"),
        ("Aripiprazole", "阿立哌唑", "N05AX12"),
        ("Sertraline", "舍曲林", "N06AB06"),
        ("Memantine", "美金刚", "N06DX01"),
        ("Duloxetine", "度洛西汀", "N06AX21"),
    ],
    "Gastrointestinal": [
        ("Omeprazole", "奥美拉唑", "A02BC01"),
        ("Esomeprazole", "埃索美拉唑", "A02BC05"),
        ("Mosapride", "莫沙必利", "A03FA09"),
        ("Lactulose", "乳果糖", "A06AD11"),
        ("Ranitidine", "雷尼替丁", "A02BA02"),
        ("Compound Digestive", "复方消化酶", "A09AA02"),
        ("Racecadotril", "消旋卡多曲", "A07XA04"),
        ("Trimebutine", "曲美布汀", "A03AA05"),
    ],
}

# 剂型和规格池
DOSAGE_FORMS = ["Tablets", "Capsules", "Granules", "Injection", "Oral Solution"]
STRENGTHS = ["5mg", "10mg", "20mg", "25mg", "50mg", "100mg", "250mg", "500mg"]
PACK_SIZES = [7, 14, 21, 28, 30, 60, 90, 100]

def parse_atc(atc_code):
    """从ATC-5编码拆解出ATC-1到ATC-4"""
    return {
        'atc_level1': atc_code[0],           # 如 C
        'atc_level2': atc_code[:3],          # 如 C08
        'atc_level3': atc_code[:4],          # 如 C08C
        'atc_level4': atc_code[:5],          # 如 C08CA
    }

# 生成500个SKU
products = []
sku_profiles = []  # 同时生成SKU画像

# 每个治疗领域分配大致均匀的SKU数量
skus_per_area = N_SKUS // len(THERAPY_AREAS)  # ~83个/领域
remaining = N_SKUS - skus_per_area * len(THERAPY_AREAS)

sku_idx = 0
for t_idx, therapy_area in enumerate(THERAPY_AREAS):
    drugs = DRUG_POOL[therapy_area]
    n_area = skus_per_area + (1 if t_idx < remaining else 0)
    
    for i in range(n_area):
        drug = drugs[i % len(drugs)]  # 循环使用药品
        en_name, cn_name, atc_code = drug
        
        # 随机生成规格
        strength = np.random.choice(STRENGTHS)
        pack_size = np.random.choice(PACK_SIZES)
        dosage_form = np.random.choice(DOSAGE_FORMS)
        
        # 构建SKU名称
        sku_name = f"{cn_name}{dosage_form} {strength}x{pack_size}"
        product_name = f"{en_name} {dosage_form} {strength}x{pack_size}"
        sku_id = f"SKU-{sku_idx+1:04d}"
        
        # 拆解ATC编码
        atc_parts = parse_atc(atc_code)
        
        # 价格（集采前价格，后续vbp_impact会算集采后价格）
        base_price = np.random.uniform(5, 80)  # 5-80元
        
        # 是否进入集采（约30%的SKU纳入集采）
        is_vbp = np.random.random() < 0.30
        
        # 保质期（月）
        shelf_life = np.random.choice([12, 18, 24, 36, 48])
        
        # 供应商
        suppliers = ["辉瑞制药", "阿斯利康", "诺华制药", "赛诺菲", "恒瑞医药",
                     "石药集团", "齐鲁制药", "扬子江药业", "正大天晴", "科伦药业"]
        supplier = np.random.choice(suppliers)
        
        # 是否需要冷链
        cold_chain = np.random.random() < 0.05  # 5%
        
        # 入库产品主数据
        products.append({
            'sku_id': sku_id,
            'atc_code': atc_code,
            'atc_level1': atc_parts['atc_level1'],
            'atc_level2': atc_parts['atc_level2'],
            'atc_level3': atc_parts['atc_level3'],
            'atc_level4': atc_parts['atc_level4'],
            'product_name': product_name,
            'sku_name': sku_name,
            'generic_name': en_name,
            'generic_name_cn': cn_name,
            'category': atc_code[:4],  # ATC-3作为品类
            'therapy_area': therapy_area,
            'dosage_form': dosage_form,
            'specification': f"{strength}x{pack_size}",
            'is_rx': 1,  # 处方药
            'is_essential_medicine': np.random.choice([0, 1], p=[0.7, 0.3]),
            'vbp_flag': int(is_vbp),
            'vbp_batch': np.random.choice([1, 2, 3, 4, 5, 7, 9]) if is_vbp else 0,
            'unit_price_cny': round(base_price, 2),
            'shelf_life_months': shelf_life,
            'storage_type': 'Cold Chain' if cold_chain else 'Ambient',
            'cold_chain': int(cold_chain),
            'supplier': supplier,
            'lead_time_days': np.random.choice([7, 14, 21, 30]),
        })
        
        # --- 同时生成SKU需求画像 ---
        
        # 决定demand_class（需求分类）
        # 比例：fast~10%, seasonal~10%, policy_shocked~2%, long_tail~78%
        rand = np.random.random()
        if rand < 0.10:
            demand_class = "fast"
            base_demand = np.random.uniform(200, 800)  # 日均销量高
            demand_cv = np.random.uniform(0.1, 0.3)    # 波动小
        elif rand < 0.20:
            demand_class = "seasonal"
            base_demand = np.random.uniform(50, 300)
            demand_cv = np.random.uniform(0.3, 0.6)
        elif rand < 0.22:
            demand_class = "policy_shocked"
            base_demand = np.random.uniform(100, 500)
            demand_cv = np.random.uniform(0.2, 0.5)
        else:
            demand_class = "long_tail"
            base_demand = np.random.uniform(1, 30)     # 日均销量极低
            demand_cv = np.random.uniform(0.8, 2.0)    # 波动极大
        
        # 是否受VBP影响
        vbp_flag_sku = int(is_vbp)
        vbp_shock_month = VBP_DATE if (is_vbp and np.random.random() < 0.5) else (VBP_DATE_2 if is_vbp else pd.NaT)
        
        sku_profiles.append({
            'sku_id': sku_id,
            'demand_class': demand_class,
            'base_demand': round(base_demand, 2),
            'demand_cv': round(demand_cv, 3),
            'cold_chain': int(cold_chain),
            'vbp_flag': vbp_flag_sku,
            'vbp_shock_month': vbp_shock_month.strftime('%Y-%m-%d') if pd.notna(vbp_shock_month) else '',
        })
        
        sku_idx += 1

# 保存products.csv
products_df = pd.DataFrame(products)
products_df.to_csv("data/products.csv", index=False)
print(f"    [OK] products.csv: {len(products_df)}行 × {len(products_df.columns)}列")

# 保存sku_profiles.csv
sku_profiles_df = pd.DataFrame(sku_profiles)
sku_profiles_df.to_csv("data/sku_profiles.csv", index=False)
print(f"    [OK] sku_profiles.csv: {len(sku_profiles_df)}行 × {len(sku_profiles_df.columns)}列")

# 统计各demand_class数量
class_counts = sku_profiles_df['demand_class'].value_counts()
print(f"    demand_class分布: {dict(class_counts)}")



# ============================================================
# 第四部分：需求引擎 — 生成500 SKU × 2333天的日度需求
# ============================================================

print(f"\n[3/8] 生成日度需求数据（{N_SKUS} SKU × {n_days}天）...")
print("    这可能需要一些时间...")

# 需求引擎核心公式（基于老师代码的mechanistic engine）：
# demand = base × trend × seasonality × flu(ILI) × vbp_shock × channel_split × noise

def generate_sku_demand(sku_row, ext_signals, dates):
    """
    为单个SKU生成完整的日度需求时间序列
    
    参数:
        sku_row: sku_profiles中的一行（含demand_class, base_demand, demand_cv等）
        ext_signals: external_signals DataFrame
        dates: 日期索引
    
    返回:
        demand_total: 全国总需求（日度数组）
        demand_hospital: 医院渠道需求
        demand_chain: 连锁药店渠道需求
        demand_independent: 独立药店渠道需求
    """
    n = len(dates)
    base = sku_row['base_demand']
    demand_class = sku_row['demand_class']
    cv = sku_row['demand_cv']
    vbp_flag = sku_row['vbp_flag']
    vbp_shock_month = sku_row['vbp_shock_month']
    
    # --- 1. 长期趋势 ---
    # 药品整体市场每年增长约5-8%（老龄化+渗透率提升）
    annual_growth = np.random.uniform(0.03, 0.08)
    trend = np.exp(np.linspace(0, annual_growth * (n / 365.25), n))
    
    # --- 2. 季节性因子 ---
    seasonal = np.ones(n)
    for i, date in enumerate(dates):
        month = date.month
        # 不同药品季节性不同
        if demand_class == 'seasonal':
            # 强季节性：冬季高夏季低（如感冒药、心血管药）
            seasonal[i] = 1.0 + 0.20 * np.sin(2 * np.pi * (month - 1) / 12)
        elif demand_class == 'fast':
            # 弱季节性（慢性病药长期稳定）
            seasonal[i] = 1.0 + 0.05 * np.sin(2 * np.pi * (month - 1) / 12)
        elif demand_class == 'policy_shocked':
            # 中等季节性
            seasonal[i] = 1.0 + 0.10 * np.sin(2 * np.pi * (month - 1) / 12)
        # long_tail: 无明显季节性（保持1.0）
    
    # --- 3. 流感活动影响 ---
    flu_factor = np.ones(n)
    flu_idx = ext_signals['flu_activity_index'].values
    for i in range(n):
        if demand_class in ['seasonal', 'fast']:
            # 流感季期间呼吸道药物和心血管药物需求略增
            flu_factor[i] = 1.0 + 0.05 * (flu_idx[i] / 10)
    
    # --- 4. VBP集采冲击 ---
    vbp_factor = np.ones(n)
    if vbp_flag and vbp_shock_month:
        vbp_date = pd.Timestamp(vbp_shock_month)
        for i, date in enumerate(dates):
            if date >= vbp_date:
                days_since = (date - vbp_date).days
                if days_since < 30:
                    # 集采后第一个月：短暂混乱（销量波动大）
                    vbp_factor[i] = np.random.uniform(0.8, 1.5)
                elif days_since < 90:
                    # 3个月内：需求快速上升（降价→更多人用）
                    vbp_factor[i] = 1.0 + (1 - np.exp(-days_since / 60)) * 0.5
                else:
                    # 3个月后：稳定在较高水平（+30~50%）
                    vbp_factor[i] = np.random.uniform(1.3, 1.6)
    
    # --- 5. 假期效应 ---
    holiday_factor = np.ones(n)
    for i in range(n):
        if ext_signals['is_holiday'].iloc[i] == 1:
            holiday_factor[i] = 0.3  # 假期销量大幅下降（药店关门）
        elif ext_signals['days_to_nearest_holiday'].iloc[i] <= 2:
            holiday_factor[i] = 1.2  # 假期前备货小高峰
    
    # --- 6. 渠道拆分 ---
    # 根据ATC编码决定各渠道占比（不同品类在不同渠道的销售比例不同）
    # 总体 = 医院 + 连锁 + 独立
    # 参考行业数据：
    #   - 处方药：医院渠道占比高（~40%）
    #   - 非处方药/慢性病药：药店渠道占比高
    #   - 连锁药店 vs 独立药店：一般6:4或7:3
    
    hospital_ratio = np.random.uniform(0.25, 0.45)  # 医院占比25-45%
    remaining = 1 - hospital_ratio
    chain_ratio = np.random.uniform(0.50, 0.70) * remaining  # 连锁占剩余部分的50-70%
    independent_ratio = remaining - chain_ratio  # 独立药店占剩余部分
    
    # --- 7. 噪声 ---
    if demand_class == 'long_tail':
        # 长尾SKU：大量0值 + 偶尔的需求爆发
        noise = np.random.poisson(base * cv, n)
        # 70%的概率为0
        zero_mask = np.random.random(n) < 0.70
        noise[zero_mask] = 0
    else:
        # 正常SKU：对数正态噪声
        noise = np.random.lognormal(0, cv, n)
    
    # --- 合成最终需求 ---
    if demand_class == 'long_tail':
        # 长尾SKU：base是均值，但大部分天是0
        demand_total = np.round(base * trend * seasonal * flu_factor * vbp_factor * holiday_factor + noise).clip(min=0)
    else:
        demand_total = np.round(base * trend * seasonal * flu_factor * vbp_factor * holiday_factor * noise).clip(min=0)
    
    # 确保整数
    demand_total = demand_total.astype(int)
    
    # 渠道拆分
    demand_hospital = np.round(demand_total * hospital_ratio).astype(int)
    demand_chain = np.round(demand_total * chain_ratio).astype(int)
    demand_independent = demand_total - demand_hospital - demand_chain  # 剩余归独立
    demand_independent = np.clip(demand_independent, 0, None)
    
    return demand_total, demand_hospital, demand_chain, demand_independent


# --- 批量生成所有SKU的需求 ---

all_demand_records = []
all_inventory_records = []
all_replenishment_records = []
all_vbp_impact_records = []

for idx, sku_row in sku_profiles_df.iterrows():
    if (idx + 1) % 50 == 0:
        print(f"    ... 已处理 {idx+1}/{N_SKUS} 个SKU")
    
    sku_id = sku_row['sku_id']
    
    # 生成该SKU的4渠道需求
    d_total, d_hosp, d_chain, d_ind = generate_sku_demand(sku_row, external_signals, date_range)
    
    # --- 生成demand_daily记录 ---
    for i, date in enumerate(date_range):
        all_demand_records.append({
            'date': date.strftime('%Y-%m-%d'),
            'sku_id': sku_id,
            'demand_total': d_total[i],
            'demand_hospital': d_hosp[i],
            'demand_chain': d_chain[i],
            'demand_independent': d_ind[i],
        })
    
    # --- 生成inventory记录（日度） ---
    # 库存逻辑：每天期末库存 = 上期库存 + 到货 - 销售
    # 到货是订货提前期前下的单
    lead_time = products[idx]['lead_time_days']
    
    inventory = np.zeros(n_days)
    receipts = np.zeros(n_days)
    stockout_flag = np.zeros(n_days, dtype=int)
    
    # 初始库存 = 60天供应量（更充足的起始库存）
    inventory[0] = int(d_total[:60].sum())
    
    # 再订货点 = lead_time期间的平均需求 + 安全库存（确保 reorder_point > safety_stock）
    avg_demand = np.mean(d_total)
    safety_stock = int(avg_demand * 30 * 0.5)
    reorder_point = int(avg_demand * lead_time + safety_stock * 1.2)
    order_qty = int(avg_demand * 45)  # 每次订45天的量（增加订货量）
    
    # 记录已下单但未到货的数量，避免重复订货
    pending_orders = 0
    
    for i in range(1, n_days):
        # 检查是否触发订货（库存+在途 < 再订货点）
        if inventory[i-1] + pending_orders < reorder_point:
            # 订货将在lead_time天后到货
            arrival_day = min(i + lead_time, n_days - 1)
            receipts[arrival_day] += order_qty
            pending_orders += order_qty
        
        # 期末库存 = 上期库存 + 到货 - 销售
        inventory[i] = max(0, inventory[i-1] + receipts[i] - d_total[i])
        
        # 如果今天有到货，减少在途订单
        if receipts[i] > 0:
            pending_orders = max(0, pending_orders - receipts[i])
        
        # 如果库存不够卖，标记缺货
        if inventory[i-1] < d_total[i]:
            stockout_flag[i] = 1
            inventory[i] = 0  # 库存清空
    
    for i, date in enumerate(date_range):
        all_inventory_records.append({
            'date': date.strftime('%Y-%m-%d'),
            'sku_id': sku_id,
            'ending_inventory': int(inventory[i]),
            'receipt_qty': int(receipts[i]),
            'stockout_flag': int(stockout_flag[i]),
        })
    
    # --- 生成replenishment记录 ---
    # 取最后30天平均库存作为当前库存（避免单天随机波动导致为0）
    current_inv = int(np.mean(inventory[-30:])) if n_days >= 30 else int(inventory[-1])
    avg_monthly = int(np.mean(d_total) * 30)
    safety_stock = int(avg_monthly * 0.5)
    reorder_pt = int(avg_monthly * (lead_time / 30 + 0.5) * 1.5)  # 确保 reorder_pt > safety_stock
    suggested_qty = max(0, reorder_pt + safety_stock - current_inv)
    
    all_replenishment_records.append({
        'sku_id': sku_id,
        'current_inventory': current_inv,
        'avg_monthly_demand': avg_monthly,
        'safety_stock': safety_stock,
        'reorder_point': reorder_pt,
        'suggested_order_qty': suggested_qty,
        'order_value_cny': round(suggested_qty * products[idx]['unit_price_cny'], 2),
        'priority': 'High' if current_inv < safety_stock else ('Medium' if current_inv < reorder_pt else 'Low'),
        'lead_time_days': lead_time,
    })
    
    # --- 生成vbp_impact记录 ---
    if sku_row['vbp_flag']:
        pre_price = products[idx]['unit_price_cny']
        post_price = round(pre_price * (1 - VBP_PRICE_DROP), 2)
        
        # 计算集采前后平均销量
        pre_vbp_demand = np.mean(d_total[date_range < VBP_DATE])
        post_vbp_demand = np.mean(d_total[date_range >= VBP_DATE])
        volume_uplift = round((post_vbp_demand / pre_vbp_demand - 1) * 100, 1) if pre_vbp_demand > 0 else 0
        
        all_vbp_impact_records.append({
            'sku_id': sku_id,
            'pre_vbp_price': pre_price,
            'post_vbp_price': post_price,
            'price_drop_pct': round(-VBP_PRICE_DROP * 100, 1),
            'volume_uplift_pct': volume_uplift,
            'vbp_batch': products[idx]['vbp_batch'],
            'vbp_date': VBP_DATE.strftime('%Y-%m-%d'),
        })


# --- 保存所有CSV文件 ---

# demand_daily.csv
demand_df = pd.DataFrame(all_demand_records)
demand_df.to_csv("data/demand_daily.csv", index=False)
print(f"    [OK] demand_daily.csv: {len(demand_df):,}行 × {len(demand_df.columns)}列")
print(f"      (500 SKU × {n_days}天 = {500*n_days:,} 理论行)")

# inventory.csv
inventory_df = pd.DataFrame(all_inventory_records)
inventory_df.to_csv("data/inventory.csv", index=False)
print(f"    [OK] inventory.csv: {len(inventory_df):,}行 × {len(inventory_df.columns)}列")

# replenishment.csv
replenishment_df = pd.DataFrame(all_replenishment_records)
replenishment_df.to_csv("data/replenishment.csv", index=False)
print(f"    [OK] replenishment.csv: {len(replenishment_df)}行 × {len(replenishment_df.columns)}列")

# vbp_impact.csv
vbp_df = pd.DataFrame(all_vbp_impact_records) if all_vbp_impact_records else pd.DataFrame()
if not vbp_df.empty:
    vbp_df.to_csv("data/vbp_impact.csv", index=False)
    print(f"    [OK] vbp_impact.csv: {len(vbp_df)}行 × {len(vbp_df.columns)}列")
else:
    print(f"    [OK] vbp_impact.csv: 空（无集采SKU）")


# ============================================================
# 第五部分：完成总结
# ============================================================

print("\n" + "=" * 60)
print("数据生成完成！输出文件列表：")
print("=" * 60)

output_files = [
    ("data/products.csv", "500 SKU主数据（含ATC-1~4拆解）"),
    ("data/demand_daily.csv", "日度需求（总体/医院/连锁/独立 4渠道）"),
    ("data/external_signals.csv", "外部信号（ILI%、百度指数、假期、VBP）"),
    ("data/holidays.csv", "中国假期日历"),
    ("data/sku_profiles.csv", "SKU需求画像分类"),
    ("data/vbp_impact.csv", "集采冲击参数"),
    ("data/inventory.csv", "日度库存快照"),
    ("data/replenishment.csv", "补货建议基线参数"),
]

for filepath, desc in output_files:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
        print(f"  [OK] {filepath:<30} {size_str:>8}  {desc}")
    else:
        print(f"  ✗ {filepath:<30}  未生成")

print("=" * 60)
print(f"\n下一步: 运行 02_preprocess.py 进行数据预处理")
print("=" * 60)
