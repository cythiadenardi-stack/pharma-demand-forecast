"""
药品需求预测与智能补货系统 - Streamlit前端
风格: 高级商务 / 大字体 / 长历史 / 精美图表
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(page_title="药品需求预测与智能补货系统", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================
# CSS - 高级商务风格 / 大字体
# ============================================================
st.markdown("""
<style>
/* 全局字体与配色 */
html, body, [class*="css"] { font-size: 18px !important; font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important; }

/* 标题 */
h1 { font-size: 38px !important; font-weight: 800 !important; color: #0f172a !important; margin-bottom: 12px !important; letter-spacing: -0.5px; }
h2 { font-size: 26px !important; font-weight: 700 !important; color: #1e293b !important; margin-top: 36px !important; margin-bottom: 16px !important; border-left: 4px solid #3b82f6; padding-left: 14px; }
h3 { font-size: 20px !important; font-weight: 600 !important; color: #334155 !important; }

/* 侧边栏 - 深色商务风 */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 16px !important; padding: 10px 0 !important; color: #e2e8f0 !important; }
[data-testid="stSidebar"] h1 { color: #f8fafc !important; }
[data-testid="stSidebar"] p { color: #94a3b8 !important; }
[data-testid="stSidebar"] hr { background: linear-gradient(90deg, transparent 0%, #475569 50%, transparent 100%) !important; }

/* 选择框 */
.stSelectbox label { font-size: 16px !important; font-weight: 600 !important; color: #374151 !important; }
[data-baseweb="select"] { font-size: 16px !important; }

/* 分隔线 */
hr { border: none; height: 2px; background: linear-gradient(90deg, #e2e8eb 0%, #3b82f6 50%, #e2e8eb 100%); margin: 32px 0; border-radius: 1px; }

/* 表格 */
[data-testid="stDataFrame"] { font-size: 15px !important; }

/* 按钮 */
.stButton>button { border-radius: 10px !important; font-size: 15px !important; padding: 10px 24px !important; font-weight: 600 !important; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; color: white !important; border: none !important; transition: all 0.2s !important; }
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important; }

/* 卡片 - 毛玻璃效果 */
.metric-container { background: rgba(255,255,255,0.95); border-radius: 16px; padding: 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); border: 1px solid rgba(226,232,240,0.8); backdrop-filter: blur(10px); transition: transform 0.2s, box-shadow 0.2s; }
.metric-container:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.1); }

/* 指标数字 */
.big-number { font-size: 40px; font-weight: 800; background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.big-label { font-size: 14px; color: #64748b; margin-top: 6px; font-weight: 500; letter-spacing: 0.3px; }

/* 警示卡片 */
.alert-box { border-radius: 12px; padding: 18px 22px; margin: 10px 0; font-size: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.alert-red { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-left: 5px solid #dc2626; }
.alert-yellow { background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%); border-left: 5px solid #eab308; }
.alert-green { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-left: 5px solid #22c55e; }

/* 信息条 */
.info-bar { background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%); color: white; 
            border-radius: 14px; padding: 18px 24px; margin: 14px 0; font-size: 16px; box-shadow: 0 4px 16px rgba(59,130,246,0.25); }

/* 标签页 */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { font-size: 15px !important; font-weight: 600; padding: 10px 20px !important; border-radius: 8px 8px 0 0; }
.stTabs [data-baseweb="tab-highlight"] { background: #3b82f6 !important; }

/* 滚动条美化 */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 翻译
# ============================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'zh'

T = {
    "zh": {
        "app_title": "药品需求预测与智能补货系统",
        "forecast": "需求预测看板", "replenish": "智能补货建议",
        "inventory": "库存健康诊断", "policy": "政策影响模拟", "alerts": "预警通知",
        "select_sku": "选择药品", "select_channel": "选择渠道",
        "total": "全国总计", "hospital": "医院", "chain": "连锁药店", "independent": "独立药店",
        "history_forecast": "历史销量与预测趋势",
        "channel_split": "渠道拆分预测", "accuracy": "预测准确度",
        "forecast_table": "90天预测明细", "feature_importance": "预测影响因素分析",
        "overview": "库存状态总览", "urgent": "紧急补货", "need": "需要补货", "sufficient": "库存充足",
        "priority_list": "补货优先级清单",
        "abc_xyz": "ABC/XYZ 分析矩阵", "near_expiry": "近效期预警", "kpi": "库存关键指标",
        "policy_sim": "集采政策影响模拟", "scenario": "选择集采场景",
        "price_drop": "降价幅度", "revenue_impact": "收入影响估算",
        "stockout_alert": "缺货风险预警", "overstock_alert": "库存积压预警",
        "sku_name": "药品名称", "current_inv": "当前库存", "safety_stock": "安全库存",
        "days_avail": "可售天数", "suggested_qty": "建议订货量",
        "order_value": "订货金额(元)", "priority": "优先级",
        "date": "日期", "quantity": "数量(盒)", "actual": "实际销量", "forecast": "预测销量",
        "loading": "正在加载数据...", "no_data": "暂无数据",
        "lang": "语言", "monthly_sales": "月销量(盒)", "cv": "变异系数",
        "shelf_life": "保质期(月)", "unit_price": "单价(元)",
        "before_vbp": "集采前", "after_vbp": "集采后",
        "price_reduction": "降幅", "volume_increase": "量增",
        "Daily Avg": "日均销量", "Avg MAPE": "平均MAPE",
        "Current Revenue": "当前月收入", "VBP Revenue": "集采后月收入", "Change": "收入变化",
        "Importance": "重要性权重",
        "All SKUs healthy": "所有药品库存健康，无断货风险",
        "No overstock": "无库存积压，库存水平正常",
        "No near-expiry": "暂无近效期药品",
        "Stock": "当前库存", "Stockout in": "预计断货天数",
        "Suggest": "建议订货", "boxes": "盒",
        "Equals": "相当于", "months supply": "个月销量",
        "Monthly Qty": "月销量(盒)", "Current Price": "当前单价(元)", "VBP Price": "集采后单价(元)",
        "High Risk": "高危", "Medium Risk": "警告",
        "history_period": "历史数据: 2025年6月 - 2026年5月 (12个月) | 预测: 2026年6月-8月 (90天)",
        "best_timing": "最佳补货时机", "est_stockout": "预计断货", "suggest_order_date": "建议下单",
        "order_now": "立即下单", "reorder_point": "再订货点", "channel_replenish": "补货渠道",
        "inventory_structure": "期末库存结构", "fast_moving": "畅销品清单", "slow_moving": "滞销品清单",
        "turnover_days": "周转天数", "inventory_value": "库存金额", "recommendation": "处理建议",
        "excess_inventory": "多余库存", "shortage": "库存缺口", "reallocation": "库存重新分配建议",
        "promotion_clearance": "促销清仓", "reduce_order": "减少订货/退货",
        "priority_replenish": "优先补货", "increase_safety_stock": "增加安全库存",
        "by_therapy_area": "按治疗领域", "by_abc_class": "按ABC分类",
        "qty_90d": "90天销量", "no_fast_moving": "暂无畅销品", "no_slow_moving": "暂无滞销品",
        "total_inventory_value": "总库存金额", "total_inventory_qty": "总库存数量",
        "healthy_skus": "健康SKU数", "slow_skus": "滞销SKU数",
    },
    "en": {
        "app_title": "Pharma Demand Forecast & Replenishment",
        "forecast": "Demand Forecast", "replenish": "Replenishment",
        "inventory": "Inventory Health", "policy": "Policy Simulator", "alerts": "Alerts",
        "select_sku": "Select Drug", "select_channel": "Select Channel",
        "total": "Total", "hospital": "Hospital", "chain": "Chain Pharmacy", "independent": "Independent",
        "history_forecast": "Historical vs Forecast",
        "channel_split": "Channel Forecast Split", "accuracy": "Forecast Accuracy",
        "forecast_table": "90-Day Forecast Detail", "feature_importance": "Prediction Factor Analysis",
        "overview": "Inventory Overview", "urgent": "Urgent", "need": "Need Replenish", "sufficient": "Sufficient",
        "priority_list": "Priority List",
        "abc_xyz": "ABC/XYZ Matrix", "near_expiry": "Near Expiry Warning", "kpi": "Inventory KPIs",
        "policy_sim": "VBP Policy Impact", "scenario": "Select VBP Scenario",
        "price_drop": "Price Reduction", "revenue_impact": "Revenue Impact",
        "stockout_alert": "Stockout Risk Alert", "overstock_alert": "Overstock Alert",
        "sku_name": "Drug Name", "current_inv": "Current Stock", "safety_stock": "Safety Stock",
        "days_avail": "Days Available", "suggested_qty": "Suggested Order",
        "order_value": "Order Value(CNY)", "priority": "Priority",
        "date": "Date", "quantity": "Qty(boxes)", "actual": "Actual", "forecast": "Forecast",
        "loading": "Loading...", "no_data": "No data",
        "lang": "Language", "monthly_sales": "Monthly Sales", "cv": "CV",
        "shelf_life": "Shelf Life(mo)", "unit_price": "Price(CNY)",
        "before_vbp": "Before VBP", "after_vbp": "After VBP",
        "price_reduction": "Reduction", "volume_increase": "Vol. Increase",
        "Daily Avg": "Daily Avg", "Avg MAPE": "Avg MAPE",
        "Current Revenue": "Current Revenue", "VBP Revenue": "VBP Revenue", "Change": "Change",
        "Importance": "Importance",
        "All SKUs healthy": "All SKUs healthy - No stockout risk",
        "No overstock": "No overstock - Inventory levels optimal",
        "No near-expiry": "No near-expiry items",
        "Stock": "Stock", "Stockout in": "Stockout in",
        "Suggest": "Suggest", "boxes": "boxes",
        "Equals": "Equals", "months supply": "months supply",
        "Monthly Qty": "Monthly Qty", "Current Price": "Current Price", "VBP Price": "VBP Price",
        "High Risk": "High Risk", "Medium Risk": "Medium Risk",
        "history_period": "History: Jun 2025 - May 2026 (12mo) | Forecast: Jun-Aug 2026 (90 days)",
        "best_timing": "Best Order Timing", "est_stockout": "Est. Stockout", "suggest_order_date": "Suggested Order",
        "order_now": "Order Now", "reorder_point": "Reorder Point", "channel_replenish": "Replenish Channel",
        "inventory_structure": "Inventory Structure", "fast_moving": "Fast-moving Items", "slow_moving": "Slow-moving Items",
        "turnover_days": "Turnover Days", "inventory_value": "Inventory Value", "recommendation": "Recommendation",
        "excess_inventory": "Excess Inventory", "shortage": "Stock Shortage", "reallocation": "Inventory Reallocation",
        "promotion_clearance": "Promote / Clear", "reduce_order": "Reduce Order / Return",
        "priority_replenish": "Priority Replenish", "increase_safety_stock": "Increase Safety Stock",
        "by_therapy_area": "By Therapy Area", "by_abc_class": "By ABC Class",
        "qty_90d": "90-Day Sales", "no_fast_moving": "No fast-moving items", "no_slow_moving": "No slow-moving items",
        "total_inventory_value": "Total Inventory Value", "total_inventory_qty": "Total Inventory Qty",
        "healthy_skus": "Healthy SKUs", "slow_skus": "Slow SKUs",
    }
}

def tr(key):
    return T[st.session_state.lang].get(key, key)


# ============================================================
# 数据加载 - 历史数据从2025年6月开始（12个月）
# ============================================================
@st.cache_data(ttl=60)
def load_all_data():
    data = {}
    files = {
        'products': 'data/products.csv',
        'replenish': 'data/replenishment.csv',
        'profiles': 'data/sku_profiles.csv',
        'vbp': 'data/vbp_impact.csv',
    }
    for key, path in files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()

    # 需求数据: 从2025年6月开始加载（12个月历史）
    if os.path.exists('data/demand_daily.csv'):
        df = pd.read_csv('data/demand_daily.csv', parse_dates=['date'])
        data['demand'] = df[df['date'] >= '2025-06-01'].copy()  # 12个月历史
    else:
        data['demand'] = pd.DataFrame()

    # 最新库存快照
    if os.path.exists('data/inventory.csv'):
        inv = pd.read_csv('data/inventory.csv', parse_dates=['date'])
        data['inventory_latest'] = inv.sort_values('date').groupby('sku_id').last().reset_index()
    else:
        data['inventory_latest'] = pd.DataFrame()

    # XGBoost预测
    if os.path.exists('results/xgboost_predictions.csv'):
        data['forecast'] = pd.read_csv('results/xgboost_predictions.csv', parse_dates=['date'])
    else:
        data['forecast'] = pd.DataFrame()

    # 指标和特征重要性
    for key, path in [('metrics', 'results/xgboost_metrics.csv'), ('fi', 'results/xgboost_feature_importance.csv')]:
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()

    return data


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    lc1, lc2 = st.columns(2)
    with lc1:
        if st.button("中文", use_container_width=True,
                     type="primary" if st.session_state.lang == 'zh' else "secondary"):
            st.session_state.lang = 'zh'; st.rerun()
    with lc2:
        if st.button("English", use_container_width=True,
                     type="primary" if st.session_state.lang == 'en' else "secondary"):
            st.session_state.lang = 'en'; st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='font-size:26px; color:#1a1a2e; line-height:1.3;'>{tr('app_title')}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; font-size:15px;'>Case 2 | Digital Innovation | 2026</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    nav_cn = ["需求预测看板", "智能补货建议", "库存健康诊断", "政策影响模拟", "预警通知"]
    nav_en = ["Demand Forecast", "Replenishment", "Inventory Health", "Policy Simulator", "Alerts"]
    nav = nav_cn if st.session_state.lang == 'zh' else nav_en
    nav_idx = st.radio("", nav, label_visibility="collapsed")
    page_idx = nav.index(nav_idx)


def metric_card(value, label, color="#0f3460"):
    return f"""
    <div class="metric-container" style="text-align:center;">
        <div class="big-number" style="color:{color};">{value}</div>
        <div class="big-label">{label}</div>
    </div>
    """


def main():
    with st.spinner(tr("loading")):
        data = load_all_data()

    if data is None or data['products'].empty:
        st.error("数据文件未找到")
        return

    products = data['products']

    # ========================================================================
    # PAGE 0: FORECAST
    # ========================================================================
    if page_idx == 0:
        st.markdown(f"<h1>{tr('forecast')}</h1>", unsafe_allow_html=True)

        forecast = data['forecast']
        demand = data['demand']
        metrics = data['metrics']
        fi = data['fi']

        if forecast.empty:
            st.warning(tr("no_data"))
            return

        # SKU选择
        sku_opts = {}
        for idx, row in products.iterrows():
            sku_opts[f"{row['sku_id']} | {row['sku_name']}"] = row['sku_id']

        c1, c2 = st.columns([3, 1])
        with c1:
            sel_label = st.selectbox(tr("select_sku"), list(sku_opts.keys()), index=0)
        sel_sku = sku_opts[sel_label]
        with c2:
            ch_opts = {tr("total"): 'total', tr("hospital"): 'hospital', tr("chain"): 'chain', tr("independent"): 'independent'}
            sel_ch_label = st.selectbox(tr("select_channel"), list(ch_opts.keys()))
        sel_ch = ch_opts[sel_ch_label]

        # SKU信息条
        info = products[products['sku_id'] == sel_sku].iloc[0]
        vbp_bg = "#d1fae5" if info['vbp_flag'] else "#e5e7eb"
        vbp_color = "#059669" if info['vbp_flag'] else "#6b7280"
        vbp_text = "VBP" if info['vbp_flag'] else "Non-VBP"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); border-radius:14px; 
                    padding:16px 24px; margin:12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.06);">
            <span style="font-size:20px; font-weight:700; color:#1a1a2e;">{info['sku_name']}</span>
            <span style="color:#6c757d; margin-left:16px; font-size:15px;">{info['sku_id']} | ATC: {info['atc_level2']}</span>
            <span style="background:{vbp_bg}; color:{vbp_color}; padding:4px 14px; border-radius:20px; 
                         font-size:13px; margin-left:16px; font-weight:600;">{vbp_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # 历史数据时间段说明
        st.markdown(f"<p style='color:#6b7280; font-size:15px; margin-bottom:16px;'>{tr('history_period')}</p>", unsafe_allow_html=True)

        # 历史+预测折线图 - 12个月历史 + 30天预测
        st.markdown(f"<h2>{tr('history_forecast')}</h2>", unsafe_allow_html=True)
        sku_fc = forecast[forecast['sku_id'] == sel_sku].sort_values('date')
        sku_hist = demand[demand['sku_id'] == sel_sku].sort_values('date') if not demand.empty else pd.DataFrame()

        try:
            import plotly.graph_objects as go
            fig = go.Figure()

            # 历史销量 - 12个月 (2025-06 到 2026-05)
            if not sku_hist.empty:
                demand_col_map = {'total': 'demand_total', 'hospital': 'demand_hospital',
                                  'chain': 'demand_chain', 'independent': 'demand_independent'}
                hist_col = demand_col_map.get(sel_ch, 'demand_total')
                if hist_col in sku_hist.columns:
                    fig.add_trace(go.Scatter(
                        x=sku_hist['date'], y=sku_hist[hist_col],
                        mode='lines', name=tr("actual"),
                        line=dict(color='#1e3a5f', width=2.5),
                        hovertemplate='%{x|%Y-%m-%d}<br>' + tr("actual") + ': %{y:.0f}<extra></extra>'
                    ))

            # 预测销量
            fig.add_trace(go.Scatter(
                x=sku_fc['date'], y=sku_fc[sel_ch],
                mode='lines+markers', name=tr("forecast"),
                line=dict(color='#c0392b', width=3, dash='dash'),
                marker=dict(size=7, color='#c0392b', symbol='diamond'),
                hovertemplate='%{x|%Y-%m-%d}<br>' + tr("forecast") + ': %{y:.0f}<extra></extra>'
            ))

            # 分界线
            if not sku_fc.empty:
                cutoff = sku_fc['date'].min()
                fig.add_vline(x=cutoff, line_dash="dot", line_color="#95a5a6", line_width=2,
                              annotation_text="历史 | 预测", annotation_position="top",
                              annotation_font_size=14)

            fig.update_layout(
                height=500, xaxis_title=tr("date"), yaxis_title=tr("quantity"),
                hovermode='x unified', font=dict(size=16, family='Arial'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                           bgcolor='rgba(255,255,255,0.9)', bordercolor='#e5e7eb', borderwidth=1),
                plot_bgcolor='#fafbfc', paper_bgcolor='white',
                margin=dict(l=70, r=40, t=80, b=50),
                xaxis=dict(gridcolor='#e5e7eb', showgrid=True),
                yaxis=dict(gridcolor='#e5e7eb', showgrid=True),
            )
            st.plotly_chart(fig, use_container_width=True,
                config=dict(scrollZoom=True, displayModeBar=True))
        except Exception as e:
            st.error(f"图表渲染出错: {str(e)}")

        # 渠道拆分 - 精美配色
        st.markdown(f"<h2>{tr('channel_split')}</h2>", unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            # 低饱和度商务配色
            colors_ch = {'hospital': '#5b7db1', 'chain': '#8db596', 'independent': '#d4a373'}
            labels_ch = {'hospital': tr('hospital'), 'chain': tr('chain'), 'independent': tr('independent')}

            for ch in ['hospital', 'chain', 'independent']:
                fig2.add_trace(go.Scatter(
                    x=sku_fc['date'], y=sku_fc[ch], name=labels_ch[ch],
                    stackgroup='one', line=dict(width=0.5, color=colors_ch[ch]),
                    fillcolor=colors_ch[ch], opacity=0.85,
                    hovertemplate='%{x|%Y-%m-%d}<br>' + labels_ch[ch] + ': %{y:.0f}<extra></extra>'
                ))

            fig2.update_layout(
                height=400, xaxis_title=tr("date"), yaxis_title=tr("quantity"),
                hovermode='x unified', font=dict(size=16, family='Arial'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                           bgcolor='rgba(255,255,255,0.9)', bordercolor='#e5e7eb', borderwidth=1),
                plot_bgcolor='#fafbfc', paper_bgcolor='white',
                margin=dict(l=70, r=40, t=80, b=50),
                xaxis=dict(gridcolor='#e5e7eb'), yaxis=dict(gridcolor='#e5e7eb'),
            )
            st.plotly_chart(fig2, use_container_width=True,
                config=dict(scrollZoom=True, displayModeBar=True))
        except Exception:
            st.bar_chart(sku_fc.set_index('date')[['hospital', 'chain', 'independent']])

        # 准确度 + 预测表
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown(f"<h2>{tr('accuracy')}</h2>", unsafe_allow_html=True)
            if not metrics.empty:
                sm = metrics[(metrics['sku_id'] == sel_sku) & (metrics['channel'] == sel_ch)]
                if not sm.empty:
                    mape_v = sm.iloc[0]['mape']
                    rmse_v = sm.iloc[0]['rmse']
                    c = '#059669' if mape_v < 20 else '#d97706' if mape_v < 40 else '#dc2626'
                    st.markdown(metric_card(f"{mape_v:.1f}%", "MAPE", c), unsafe_allow_html=True)
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    st.markdown(metric_card(f"{rmse_v:.1f}", "RMSE (盒)", "#16213e"), unsafe_allow_html=True)

        with col_b:
            st.markdown(f"<h2>{tr('forecast_table')}</h2>", unsafe_allow_html=True)
            disp = sku_fc[['date', 'total', 'hospital', 'chain', 'independent']].copy()
            disp.columns = [tr("date"), tr("total"), tr("hospital"), tr("chain"), tr("independent")]
            disp[tr("date")] = disp[tr("date")].dt.strftime('%Y-%m-%d')
            st.dataframe(disp, use_container_width=True, hide_index=True, height=380)

        # 特征重要性
        if not fi.empty:
            st.markdown(f"<h2>{tr('feature_importance')}</h2>", unsafe_allow_html=True)
            fn_map_zh = {
                'is_holiday': '是否假期', 'is_post_vbp': '集采后标记', 'days_since_vbp_log': '集采后天数',
                'rolling_mean_30': '近30天平均销量', 'baidu_flu_index': '百度流感搜索指数',
                'month_sin': '月份周期(正弦)', 'flu_x_vbp': '流感x集采交互',
                'is_weekend': '是否周末', 'flu_activity_index': '流感活动指数',
                'lag_7': '7天前销量', 'lag_14': '14天前销量', 'lag_30': '30天前销量',
                'rolling_mean_7': '近7天平均销量', 'rolling_std_7': '近7天销量波动',
                'dow_sin': '周几周期(正弦)', 'dow_cos': '周几周期(余弦)',
                'month_cos': '月份周期(余弦)', 'base_demand': '基础需求量',
                'demand_cv': '需求波动系数', 'vbp_flag': '是否集采',
                'is_flu_season': '是否流感季'
            }
            fn_map_en = {
                'is_holiday': 'Is Holiday', 'is_post_vbp': 'Post-VBP Flag', 'days_since_vbp_log': 'Days Since VBP (log)',
                'rolling_mean_30': 'Rolling Mean (30d)', 'baidu_flu_index': 'Baidu Flu Index',
                'month_sin': 'Monthly Cycle (sin)', 'flu_x_vbp': 'Flu x VBP Interaction',
                'is_weekend': 'Is Weekend', 'flu_activity_index': 'Flu Activity Index',
                'lag_7': 'Sales 7d Ago', 'lag_14': 'Sales 14d Ago', 'lag_30': 'Sales 30d Ago',
                'rolling_mean_7': 'Rolling Mean (7d)', 'rolling_std_7': 'Rolling Std (7d)',
                'dow_sin': 'Day-of-Week (sin)', 'dow_cos': 'Day-of-Week (cos)',
                'month_cos': 'Monthly Cycle (cos)', 'base_demand': 'Base Demand',
                'demand_cv': 'Demand CV', 'vbp_flag': 'VBP Flag',
                'is_flu_season': 'Is Flu Season'
            }
            fn_map = fn_map_zh if st.session_state.lang == 'zh' else fn_map_en
            fi_d = fi.copy()
            fi_d['name'] = fi_d['feature'].map(fn_map).fillna(fi_d['feature'])
            try:
                import plotly.graph_objects as go
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=fi_d['importance'], y=fi_d['name'],
                    orientation='h', marker=dict(color='#1e3a5f', line=dict(color='#152a45', width=1))
                ))
                fig3.update_layout(
                    height=500, xaxis_title=tr("Importance"), yaxis_title="",
                    yaxis=dict(autorange="reversed", tickfont=dict(size=15), gridcolor='#e5e7eb'),
                    font=dict(size=16, family='Arial'),
                    plot_bgcolor='#fafbfc', paper_bgcolor='white',
                    margin=dict(l=160, r=40, t=40, b=50),
                    xaxis=dict(gridcolor='#e5e7eb'),
                )
                st.plotly_chart(fig3, use_container_width=True,
                    config=dict(scrollZoom=True, displayModeBar=True))
            except Exception:
                st.bar_chart(fi_d.set_index('name')['importance'])

    # ========================================================================
    # PAGE 1: REPLENISHMENT
    # ========================================================================
    elif page_idx == 1:
        st.markdown(f"<h1>{tr('replenish')}</h1>", unsafe_allow_html=True)
        repl = data['replenish']
        demand_all = data['demand']
        if repl.empty:
            st.warning(tr("no_data")); return

        # 渠道选择器
        ch_opts_label = {tr("total"): 'total', tr("hospital"): 'hospital',
                         tr("chain"): 'chain', tr("independent"): 'independent'}
        sel_ch_label = st.selectbox(tr("channel_replenish"), list(ch_opts_label.keys()), index=0)
        sel_channel = ch_opts_label[sel_ch_label]

        # 合并产品和渠道数据 (replenishment.csv 已含 lead_time_days，无需重复合并)
        repl_m = repl.merge(products[['sku_id', 'sku_name', 'therapy_area', 'unit_price_cny']],
                            on='sku_id', how='left')

        # 计算各渠道月均需求及占比
        if not demand_all.empty:
            recent_demand = demand_all[demand_all['date'] >= '2026-04-01']
            channel_monthly = recent_demand.groupby('sku_id')[['demand_hospital', 'demand_chain', 'demand_independent']].mean() * 30
            channel_monthly['total'] = channel_monthly.sum(axis=1)
            for ch in ['hospital', 'chain', 'independent']:
                channel_monthly[f'ratio_{ch}'] = channel_monthly[f'demand_{ch}'] / channel_monthly['total']
            repl_m = repl_m.merge(channel_monthly.reset_index(), on='sku_id', how='left')
        else:
            for col in ['demand_hospital', 'demand_chain', 'demand_independent', 'total',
                        'ratio_hospital', 'ratio_chain', 'ratio_independent']:
                repl_m[col] = 0

        TODAY = pd.Timestamp('2026-05-31')

        # 按渠道重新计算补货参数
        if sel_channel != 'total':
            ch_col = f'demand_{sel_channel}'
            ratio_col = f'ratio_{sel_channel}'
            repl_m['avg_monthly_demand'] = repl_m[ch_col].fillna(0)
            repl_m['current_inventory'] = (repl_m['current_inventory'] * repl_m[ratio_col].fillna(0)).round().astype(int)
            repl_m['safety_stock'] = ((repl_m['avg_monthly_demand'] / 30 * 15).round().astype(int)).clip(lower=0)
            repl_m['reorder_point'] = ((repl_m['avg_monthly_demand'] / 30 * repl_m['lead_time_days'] * 1.5).round().astype(int)).clip(lower=0)
            repl_m['suggested_order_qty'] = (((repl_m['avg_monthly_demand'] / 30) * (repl_m['lead_time_days'] + 14) - repl_m['current_inventory']).clip(lower=0)).round().astype(int)
            repl_m['order_value_cny'] = (repl_m['suggested_order_qty'] * repl_m['unit_price_cny']).round(2)

        # 可售天数
        repl_m['daily'] = repl_m['avg_monthly_demand'] / 30
        repl_m['days'] = (repl_m['current_inventory'] / repl_m['daily'].replace(0, 0.1)).round(1)

        # 最佳补货时机
        repl_m['est_stockout_date'] = TODAY + pd.to_timedelta(repl_m['days'].clip(lower=0), unit='D')
        repl_m['suggest_order_date'] = repl_m['est_stockout_date'] - pd.to_timedelta(repl_m['lead_time_days'].fillna(0), unit='D')

        def fmt_timing(row, lang):
            if row['suggest_order_date'] <= TODAY:
                return '立即下单' if lang == 'zh' else 'Order Now'
            dd = (row['suggest_order_date'] - TODAY).days
            ds = row['suggest_order_date'].strftime('%m-%d')
            return f"{ds} ({dd}天后)" if lang == 'zh' else f"{ds} ({dd}d)"

        repl_m['timing_display'] = repl_m.apply(lambda r: fmt_timing(r, st.session_state.lang), axis=1)

        # 优先级计算
        def calc_priority(row):
            inv = row['current_inventory']
            ss = row['safety_stock']
            rp = row['reorder_point']
            if inv == 0:
                return 'High'
            elif inv < ss:
                return 'High'
            elif inv < rp:
                return 'Medium'
            else:
                return 'Low'

        repl_m['priority'] = repl_m.apply(calc_priority, axis=1)

        # 中文模式翻译
        if st.session_state.lang == 'zh':
            ta_map = {
                'Cardiovascular': '心血管', 'Antibiotics': '抗生素', 'Diabetes': '糖尿病',
                'CNS': '中枢神经系统', 'Respiratory': '呼吸系统', 'Gastrointestinal': '消化系统'
            }
            repl_m['therapy_area'] = repl_m['therapy_area'].map(ta_map).fillna(repl_m['therapy_area'])
            pri_map = {'High': '紧急', 'Medium': '需要补货', 'Low': '库存充足'}
            repl_m['priority'] = repl_m['priority'].map(pri_map)
            h = (repl_m['priority'] == '紧急').sum()
            m = (repl_m['priority'] == '需要补货').sum()
            l = (repl_m['priority'] == '库存充足').sum()
        else:
            h = (repl_m['priority'] == 'High').sum()
            m = (repl_m['priority'] == 'Medium').sum()
            l = (repl_m['priority'] == 'Low').sum()

        c1, c2, c3 = st.columns(3)
        for col, val, label, tc, bc, icon in [
            (c1, h, tr("urgent"), '#dc2626', '#fef2f2', '🔴'),
            (c2, m, tr("need"), '#d97706', '#fefce8', '🟡'),
            (c3, l, tr("sufficient"), '#059669', '#f0fdf4', '🟢')
        ]:
            with col:
                st.markdown(f"""
                <div style="background:{bc}; border-radius:16px; padding:28px; text-align:center;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 2px solid {tc}22;
                            transition: transform 0.2s;">
                    <div style="font-size:36px; margin-bottom:4px;">{icon}</div>
                    <div style="font-size:44px; font-weight:800; color:{tc};">{val}</div>
                    <div style="font-size:17px; color:#374151; margin-top:8px; font-weight:600;">{label}</div>
                    <div style="font-size:13px; color:#9ca3af;">SKU</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h2>{tr('priority_list')}</h2>", unsafe_allow_html=True)

        # 筛选器
        filt_cols = st.columns([2, 2, 3])
        with filt_cols[0]:
            if st.session_state.lang == 'zh':
                pri_opts = ['紧急', '需要补货', '库存充足']
            else:
                pri_opts = ['High', 'Medium', 'Low']
            pri_filter = st.multiselect("优先级筛选", pri_opts, default=pri_opts)
        with filt_cols[1]:
            therapy_areas = sorted(repl_m['therapy_area'].dropna().unique())
            ta_filter = st.multiselect("治疗领域", therapy_areas, default=therapy_areas)
        with filt_cols[2]:
            search_term = st.text_input("搜索药品名称", placeholder="输入关键词...")

        filtered = repl_m[repl_m['priority'].isin(pri_filter) & repl_m['therapy_area'].isin(ta_filter)]
        if search_term:
            filtered = filtered[filtered['sku_name'].str.contains(search_term, case=False, na=False)]

        st.markdown(f"<p style='color:#64748b; font-size:14px;'>显示 {len(filtered)} / {len(repl_m)} 个 SKU</p>", unsafe_allow_html=True)

        disp = filtered[['priority', 'sku_name', 'therapy_area', 'current_inventory', 'safety_stock',
                         'days', 'suggested_order_qty', 'order_value_cny', 'timing_display']].head(200)
        disp.columns = [tr("priority"), tr("sku_name"), '治疗领域', tr("current_inv"), tr("safety_stock"),
                        tr("days_avail"), tr("suggested_qty"), tr("order_value"), tr("best_timing")]

        def color_pri(v):
            if st.session_state.lang == 'zh':
                if v == '紧急': return 'background-color:#fef2f2; color:#dc2626; font-weight:700; border-radius:6px; padding:4px 12px;'
                if v == '需要补货': return 'background-color:#fefce8; color:#d97706; font-weight:700; border-radius:6px; padding:4px 12px;'
                return 'background-color:#f0fdf4; color:#059669; font-weight:700; border-radius:6px; padding:4px 12px;'
            else:
                if v == 'High': return 'background-color:#fef2f2; color:#dc2626; font-weight:700; border-radius:6px; padding:4px 12px;'
                if v == 'Medium': return 'background-color:#fefce8; color:#d97706; font-weight:700; border-radius:6px; padding:4px 12px;'
                return 'background-color:#f0fdf4; color:#059669; font-weight:700; border-radius:6px; padding:4px 12px;'

        def color_inv(v):
            if v == 0: return 'color:#dc2626; font-weight:700;'
            elif v < 100: return 'color:#d97706; font-weight:600;'
            return ''

        def color_timing(v):
            if '立即' in str(v) or 'Now' in str(v):
                return 'color:#dc2626; font-weight:700;'
            return ''

        styled = (disp.style
            .map(color_pri, subset=[tr("priority")])
            .map(color_inv, subset=[tr("current_inv")])
            .map(color_timing, subset=[tr("best_timing")]))
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # ========================================================================
    # PAGE 2: INVENTORY HEALTH
    # ========================================================================
    elif page_idx == 2:
        st.markdown(f"<h1>{tr('inventory')}</h1>", unsafe_allow_html=True)
        demand = data['demand']
        inv_latest = data.get('inventory_latest', pd.DataFrame())

        if demand.empty or inv_latest.empty:
            st.warning(tr("no_data"))
            return

        # ============================================================
        # 1. 基础计算：ABC分类 + 最新库存
        # ============================================================
        recent = demand[demand['date'] >= '2026-03-01']
        stats = recent.groupby('sku_id')['demand_total'].agg(['sum', 'mean', 'std']).reset_index()
        stats.columns = ['sku_id', 'total_qty', 'avg_daily', 'std_daily']
        stats['cv'] = (stats['std_daily'] / stats['avg_daily'].replace(0, 0.1)).fillna(0).clip(0, 5)
        stats = stats.merge(products[['sku_id', 'unit_price_cny', 'sku_name', 'therapy_area', 'shelf_life_months']], on='sku_id')
        stats['revenue'] = stats['total_qty'] * stats['unit_price_cny']
        stats = stats.sort_values('revenue', ascending=False).reset_index(drop=True)
        stats['cum_pct'] = stats['revenue'].cumsum() / stats['revenue'].sum() * 100
        stats['abc'] = stats['cum_pct'].apply(lambda x: 'A' if x <= 80 else 'B' if x <= 95 else 'C')
        stats['xyz'] = stats['cv'].apply(lambda x: 'X' if x < 0.5 else 'Y' if x < 1.0 else 'Z')
        stats['abc_xyz'] = stats['abc'] + stats['xyz']

        # 最新库存
        inv = inv_latest.merge(products[['sku_id', 'sku_name', 'unit_price_cny', 'therapy_area', 'shelf_life_months']], on='sku_id')
        inv['inventory_value'] = inv['ending_inventory'] * inv['unit_price_cny']
        inv_stats = stats[['sku_id', 'abc', 'abc_xyz', 'cv', 'revenue']].merge(inv, on='sku_id', how='right')

        # 中文翻译治疗领域
        if st.session_state.lang == 'zh':
            ta_map = {
                'Cardiovascular': '心血管', 'Antibiotics': '抗生素', 'Diabetes': '糖尿病',
                'CNS': '中枢神经系统', 'Respiratory': '呼吸系统', 'Gastrointestinal': '消化系统'
            }
            inv_stats['therapy_area'] = inv_stats['therapy_area'].map(ta_map).fillna(inv_stats['therapy_area'])
            stats['therapy_area'] = stats['therapy_area'].map(ta_map).fillna(stats['therapy_area'])
            inv['therapy_area'] = inv['therapy_area'].map(ta_map).fillna(inv['therapy_area'])

        # ============================================================
        # 2. 期末库存结构分析
        # ============================================================
        st.markdown(f"<h2>{tr('inventory_structure')}</h2>", unsafe_allow_html=True)

        # 指标卡
        total_value = inv_stats['inventory_value'].sum()
        total_qty = inv_stats['ending_inventory'].sum()
        healthy_count = len(inv_stats[inv_stats['abc_xyz'].isin(['AX', 'AY', 'BX', 'BY'])])
        slow_count = len(inv_stats[inv_stats['abc_xyz'].isin(['CZ', 'CY'])])

        kpi_data = [
            (f"¥{total_value:,.0f}", tr("total_inventory_value"), '#1e3a5f'),
            (f"{total_qty:,.0f}", tr("total_inventory_qty"), '#2c5282'),
            (f"{healthy_count}", tr("healthy_skus"), '#059669'),
            (f"{slow_count}", tr("slow_skus"), '#d97706')
        ]
        cols = st.columns(4)
        for col, (v, l, c) in zip(cols, kpi_data):
            with col:
                st.markdown(metric_card(v, l, c), unsafe_allow_html=True)

        # 饼图
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            abc_summary = inv_stats.groupby('abc').agg({
                'inventory_value': 'sum', 'ending_inventory': 'sum'
            }).reset_index()
            ta_summary = inv_stats.groupby('therapy_area').agg({
                'inventory_value': 'sum'
            }).reset_index().sort_values('inventory_value', ascending=False)

            fig_struct = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                       subplot_titles=[tr("by_abc_class"), tr("by_therapy_area")])
            fig_struct.add_trace(go.Pie(labels=abc_summary['abc'], values=abc_summary['inventory_value'],
                                         name=tr("inventory_value"), hole=.4,
                                         marker_colors=['#1e3a5f', '#5b7db1', '#d4a373']), 1, 1)
            fig_struct.add_trace(go.Pie(labels=ta_summary['therapy_area'], values=ta_summary['inventory_value'],
                                         name=tr("inventory_value"), hole=.4), 1, 2)
            fig_struct.update_layout(height=420, font=dict(size=14), showlegend=True,
                                      plot_bgcolor='#fafbfc', paper_bgcolor='white')
            st.plotly_chart(fig_struct, use_container_width=True,
                config=dict(scrollZoom=True, displayModeBar=True))
        except Exception as e:
            st.error(f"图表渲染出错: {str(e)}")

        # ============================================================
        # 3. ABC/XYZ 矩阵
        # ============================================================
        st.markdown(f"<h2>{tr('abc_xyz')}</h2>", unsafe_allow_html=True)
        try:
            import plotly.express as px
            cm = {'AX': '#1e3a5f', 'AY': '#4a6fa5', 'AZ': '#8db596',
                  'BX': '#5b7db1', 'BY': '#7b9fd1', 'BZ': '#a8c5e2',
                  'CX': '#d4a373', 'CY': '#e8c39e', 'CZ': '#f0e2cc'}
            fig = px.scatter(stats, x='revenue', y='cv', color='abc_xyz', color_discrete_map=cm,
                             size='avg_daily', hover_data=['sku_name'], height=500,
                             labels={'revenue': tr("monthly_sales"), 'cv': tr("cv")})
            fig.update_layout(font=dict(size=16), plot_bgcolor='#fafbfc', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True,
                config=dict(scrollZoom=True, displayModeBar=True))
        except Exception:
            st.scatter_chart(stats.set_index('sku_id')[['revenue', 'cv']])

        # ============================================================
        # 4. 畅销品 & 滞销品识别
        # ============================================================
        recent90 = demand[demand['date'] >= '2026-03-01']
        sales90 = recent90.groupby('sku_id')['demand_total'].agg(['sum', 'mean']).reset_index()
        sales90.columns = ['sku_id', 'qty_90d', 'avg_daily_90d']

        inv_sales = inv.merge(sales90, on='sku_id', how='left')
        inv_sales['avg_daily_90d'] = inv_sales['avg_daily_90d'].fillna(0)
        inv_sales['qty_90d'] = inv_sales['qty_90d'].fillna(0)
        inv_sales['turnover_days'] = (inv_sales['ending_inventory'] / inv_sales['avg_daily_90d'].replace(0, 0.1)).round(0).astype(int)
        inv_sales['inventory_value'] = inv_sales['ending_inventory'] * inv_sales['unit_price_cny']

        # 畅销品
        st.markdown(f"<h2>{tr('fast_moving')}</h2>", unsafe_allow_html=True)
        fast = inv_sales[(inv_sales['avg_daily_90d'] >= 10) & (inv_sales['turnover_days'] < 60)].copy()
        fast = fast.sort_values('avg_daily_90d', ascending=False)
        if not fast.empty:
            fast['recommendation'] = tr("priority_replenish") + " / " + tr("increase_safety_stock")
            fast_disp = fast[['sku_name', 'therapy_area', 'ending_inventory', 'qty_90d', 'turnover_days', 'inventory_value', 'recommendation']].head(30)
            fast_disp.columns = [tr("sku_name"), '治疗领域', tr("current_inv"), tr("qty_90d"),
                                 tr("turnover_days"), tr("inventory_value"), tr("recommendation")]
            st.dataframe(fast_disp, use_container_width=True, hide_index=True, height=350)
        else:
            st.info(tr("no_fast_moving"))

        # 滞销品
        st.markdown(f"<h2>{tr('slow_moving')}</h2>", unsafe_allow_html=True)
        slow = inv_sales[(inv_sales['avg_daily_90d'] < 1) & (inv_sales['turnover_days'] > 180)].copy()
        slow = slow.sort_values('turnover_days', ascending=False)
        if not slow.empty:
            def slow_rec(x, lang):
                if x > 360:
                    return tr("promotion_clearance")
                return tr("reduce_order")
            slow['recommendation'] = slow['turnover_days'].apply(lambda x: slow_rec(x, st.session_state.lang))
            slow_disp = slow[['sku_name', 'therapy_area', 'ending_inventory', 'qty_90d', 'turnover_days', 'inventory_value', 'recommendation']].head(30)
            slow_disp.columns = [tr("sku_name"), '治疗领域', tr("current_inv"), tr("qty_90d"),
                                 tr("turnover_days"), tr("inventory_value"), tr("recommendation")]
            st.dataframe(slow_disp, use_container_width=True, hide_index=True, height=350)
        else:
            st.info(tr("no_slow_moving"))

        # ============================================================
        # 5. 库存重新分配建议
        # ============================================================
        st.markdown(f"<h2>{tr('reallocation')}</h2>", unsafe_allow_html=True)
        if not slow.empty or not fast.empty:
            slow['excess_inventory'] = (slow['ending_inventory'] - slow['avg_daily_90d'] * 90 * 1.5).clip(lower=0).round().astype(int)
            slow['excess_value'] = slow['excess_inventory'] * slow['unit_price_cny']
            fast['shortage'] = (fast['avg_daily_90d'] * 30 - fast['ending_inventory']).clip(lower=0).round().astype(int)
            fast['shortage_value'] = fast['shortage'] * fast['unit_price_cny']

            total_excess_qty = slow['excess_inventory'].sum()
            total_excess_val = slow['excess_value'].sum()
            total_shortage_qty = fast['shortage'].sum()
            total_shortage_val = fast['shortage_value'].sum()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(metric_card(f"{total_excess_qty:,.0f}盒", tr("excess_inventory"), "#d97706"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card(f"¥{total_excess_val:,.0f}", tr("inventory_value"), "#1e3a5f"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card(f"{total_shortage_qty:,.0f}盒", tr("shortage"), "#dc2626"), unsafe_allow_html=True)

            if total_excess_qty > 0 and total_shortage_qty > 0:
                if st.session_state.lang == 'zh':
                    st.markdown(f"""
                    <div class="info-bar">
                        💡 建议：从 <b>{len(slow[slow['excess_inventory'] > 0])}</b> 个滞销品调出
                        <b>{total_excess_qty:,.0f} 盒</b> 库存，优先补充到 <b>{len(fast[fast['shortage'] > 0])}</b> 个畅销品，
                        预计可满足畅销品 <b>{min(total_excess_qty / total_shortage_qty * 100, 100):.0f}%</b> 的缺口。
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-bar">
                        💡 Suggestion: Reallocate <b>{total_excess_qty:,.0f} boxes</b> from
                        <b>{len(slow[slow['excess_inventory'] > 0])}</b> slow-moving SKUs to
                        <b>{len(fast[fast['shortage'] > 0])}</b> fast-moving SKUs, covering approximately
                        <b>{min(total_excess_qty / total_shortage_qty * 100, 100):.0f}%</b> of the shortage.
                    </div>
                    """, unsafe_allow_html=True)

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(f"<h4>{tr('slow_moving')} — {tr('excess_inventory')}</h4>", unsafe_allow_html=True)
                    slow_realloc = slow[slow['excess_inventory'] > 0][['sku_name', 'therapy_area', 'excess_inventory', 'excess_value']].head(10)
                    slow_realloc.columns = [tr("sku_name"), '治疗领域', tr("excess_inventory"), tr("inventory_value")]
                    st.dataframe(slow_realloc, use_container_width=True, hide_index=True, height=280)
                with col_r:
                    st.markdown(f"<h4>{tr('fast_moving')} — {tr('shortage')}</h4>", unsafe_allow_html=True)
                    fast_realloc = fast[fast['shortage'] > 0][['sku_name', 'therapy_area', 'shortage', 'shortage_value']].head(10)
                    fast_realloc.columns = [tr("sku_name"), '治疗领域', tr("shortage"), tr("inventory_value")]
                    st.dataframe(fast_realloc, use_container_width=True, hide_index=True, height=280)
            else:
                st.info("库存结构合理，无需大规模重新分配" if st.session_state.lang == 'zh' else "Inventory structure is balanced; no major reallocation needed.")
        else:
            st.info("库存结构合理，无需大规模重新分配" if st.session_state.lang == 'zh' else "Inventory structure is balanced; no major reallocation needed.")

        # ============================================================
        # 6. 近效期预警（增强版）
        # ============================================================
        st.markdown(f"<h2>{tr('near_expiry')}</h2>", unsafe_allow_html=True)
        expiry = inv[inv['shelf_life_months'] <= 12].copy()
        if not expiry.empty:
            expiry = expiry.merge(sales90, on='sku_id', how='left')
            expiry['avg_daily_90d'] = expiry['avg_daily_90d'].fillna(0)
            expiry['turnover_days'] = (expiry['ending_inventory'] / expiry['avg_daily_90d'].replace(0, 0.1)).round(0).astype(int)
            expiry['inventory_value'] = expiry['ending_inventory'] * expiry['unit_price_cny']
            expiry['risk'] = expiry['shelf_life_months'].apply(lambda x: tr('High Risk') if x <= 6 else tr('Medium Risk'))

            def expiry_rec(row, lang):
                if row['turnover_days'] > row['shelf_life_months'] * 30:
                    return '紧急处理/退货' if lang == 'zh' else 'Urgent / Return'
                elif row['turnover_days'] > row['shelf_life_months'] * 15:
                    return '加快销售/调拨' if lang == 'zh' else 'Accelerate / Transfer'
                return '监控效期' if lang == 'zh' else 'Monitor'

            expiry['recommendation'] = expiry.apply(lambda r: expiry_rec(r, st.session_state.lang), axis=1)
            expiry_disp = expiry[['sku_name', 'shelf_life_months', 'ending_inventory', 'turnover_days', 'inventory_value', 'risk', 'recommendation']].sort_values('shelf_life_months')
            expiry_disp.columns = [tr("sku_name"), tr("shelf_life"), tr("current_inv"), tr("turnover_days"),
                                   tr("inventory_value"), tr("priority"), tr("recommendation")]
            st.dataframe(expiry_disp, use_container_width=True, hide_index=True, height=350)
        else:
            st.info(tr("No near-expiry"))

    # ========================================================================
    # PAGE 3: POLICY
    # ========================================================================
    elif page_idx == 3:
        st.markdown(f"<h1>{tr('policy')}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#6b7280; font-size:16px;'>{tr('policy_sim')}</p>", unsafe_allow_html=True)
        vbp = data['vbp']
        if vbp.empty:
            st.warning(tr("no_data")); return

        c1, c2, c3 = st.columns(3)
        with c1:
            batch = st.selectbox(tr("scenario"), sorted(vbp['vbp_batch'].unique()))
        with c2:
            pr_drop = st.slider(tr("price_drop"), 0, 80, 51) / 100
        with c3:
            vol_up = st.slider(tr("volume_increase"), 0, 100, 45) / 100

        batch_df = vbp[vbp['vbp_batch'] == batch]
        if not batch_df.empty:
            avg_pre = batch_df['pre_vbp_price'].mean()
            avg_post = batch_df['post_vbp_price'].mean()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(metric_card(f"Y{avg_pre:.1f}", tr("before_vbp"), "#1e3a5f"), unsafe_allow_html=True)
            with m2:
                st.markdown(metric_card(f"Y{avg_post:.1f}", tr("after_vbp"), "#8b4557"), unsafe_allow_html=True)
            with m3:
                pct = (1 - avg_post / avg_pre) * 100
                st.markdown(metric_card(f"-{pct:.0f}%", tr("price_reduction"), "#c0392b"), unsafe_allow_html=True)

            st.markdown(f"<h2>{tr('revenue_impact')}</h2>", unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                mq = st.number_input(tr("Monthly Qty"), 0, 100000, 1000, 100)
            with cc2:
                cp = st.number_input(tr("Current Price"), 0.0, 1000.0, float(avg_pre), 1.0)
            with cc3:
                np_val = round(cp * (1 - pr_drop), 2)
                np_in = st.number_input(tr("VBP Price"), 0.0, 1000.0, np_val, 1.0)

            cr = mq * cp
            nr = int(mq * (1 + vol_up)) * np_in
            chg = nr - cr

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(metric_card(f"Y{cr:,.0f}", tr("Current Revenue"), "#1e3a5f"), unsafe_allow_html=True)
            with r2:
                st.markdown(metric_card(f"Y{nr:,.0f}", tr("VBP Revenue"), "#8b4557"), unsafe_allow_html=True)
            with r3:
                cc = '#059669' if chg > 0 else '#c0392b'
                st.markdown(metric_card(f"Y{chg:,.0f}", f"{chg/cr*100:.1f}%", cc), unsafe_allow_html=True)

    # ========================================================================
    # PAGE 4: ALERTS
    # ========================================================================
    elif page_idx == 4:
        st.markdown(f"<h1>{tr('alerts')}</h1>", unsafe_allow_html=True)
        repl = data['replenish']
        if repl.empty:
            st.warning(tr("no_data")); return

        repl_n = repl.merge(products[['sku_id', 'sku_name']], on='sku_id', how='left')
        repl_n['daily'] = repl_n['avg_monthly_demand'] / 30
        repl_n['days_to_stockout'] = (repl_n['current_inventory'] / repl_n['daily'].replace(0, 0.1)).round(0)

        st.markdown(f"<h2>{tr('stockout_alert')}</h2>", unsafe_allow_html=True)
        stockout = repl_n[repl_n['priority'] == 'High'].sort_values('days_to_stockout').head(15)
        if not stockout.empty:
            for idx, row in stockout.iterrows():
                st.markdown(f"""
                <div class="alert-box alert-red">
                    <span style="font-weight:700; font-size:17px;">{row['sku_name']}</span>
                    <span style="color:#6b7280; margin-left:12px; font-size:14px;">{row['sku_id']}</span><br>
                    <span style="font-size:15px; color:#4b5563;">
                    {tr('Stock')}: <b>{row['current_inventory']} {tr('boxes')}</b> |
                    {tr('Stockout in')}: <b>{row['days_to_stockout']:.0f} {tr('date')}</b> |
                    {tr('Suggest')}: <b>{row['suggested_order_qty']} {tr('boxes')}</b>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-box alert-green" style="text-align:center;"><span style="font-size:17px; font-weight:600;">{tr("All SKUs healthy")}</span></div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"<h2>{tr('overstock_alert')}</h2>", unsafe_allow_html=True)
        repl_n['month_ratio'] = repl_n['current_inventory'] / repl_n['avg_monthly_demand'].replace(0, 0.1)
        overstock = repl_n[repl_n['month_ratio'] > 3].sort_values('month_ratio', ascending=False).head(15)
        if not overstock.empty:
            for idx, row in overstock.iterrows():
                st.markdown(f"""
                <div class="alert-box alert-yellow">
                    <span style="font-weight:700; font-size:17px;">{row['sku_name']}</span>
                    <span style="color:#6b7280; margin-left:12px; font-size:14px;">{row['sku_id']}</span><br>
                    <span style="font-size:15px; color:#4b5563;">
                    {tr('Stock')}: <b>{row['current_inventory']} {tr('boxes')}</b> |
                    {tr('Equals')}: <b>{row['month_ratio']:.1f} {tr('months supply')}</b>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-box alert-green" style="text-align:center;"><span style="font-size:17px; font-weight:600;">{tr("No overstock")}</span></div>', unsafe_allow_html=True)

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center; color:#9ca3af; font-size:14px;">
        Case 2: Pharmaceutical Demand Forecasting & Intelligent Replenishment |
        SDC MSc Innovation Management - Digital Innovation | 2026
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
