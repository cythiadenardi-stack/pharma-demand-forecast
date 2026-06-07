"""
模型自动路由 — 06_model_router.py
===================================
一键运行所有4种预测模型：
- ETS → fast SKU
- Prophet → seasonal SKU
- Croston → long_tail SKU
- XGBoost → policy_shocked SKU

根据 sku_profiles.csv 中的 demand_class 字段自动路由。

运行: python 06_model_router.py
"""

import os
import subprocess
import time

print("=" * 60)
print("药品需求预测 — 模型自动路由")
print("=" * 60)

# 模型配置：按demand_class路由到对应脚本
MODELS = [
    {
        'name': 'ETS指数平滑',
        'script': '03_model_ets.py',
        'target': 'fast SKU（高销量、低波动）',
    },
    {
        'name': 'Prophet季节性分解',
        'script': '03_model_prophet.py',
        'target': 'seasonal SKU（季节性波动）',
    },
    {
        'name': 'Croston间歇需求',
        'script': '03_model_croston.py',
        'target': 'long_tail SKU（间歇性需求）',
    },
    {
        'name': 'XGBoost机器学习',
        'script': '03_model_xgboost.py',
        'target': 'policy_shocked SKU（政策冲击）',
    },
]

# 创建结果目录
os.makedirs("results", exist_ok=True)

# 依次运行每个模型
results = []
for i, model in enumerate(MODELS, 1):
    print(f"\n[{i}/{len(MODELS)}] 运行 {model['name']}...")
    print(f"     目标: {model['target']}")
    print(f"     脚本: {model['script']}")
    
    start_time = time.time()
    
    try:
        # 运行模型脚本
        result = subprocess.run(
            ['python', model['script']],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"     ✅ 成功 ({elapsed:.1f}秒)")
            results.append({'name': model['name'], 'status': '成功', 'time': elapsed})
        else:
            print(f"     ❌ 失败 (退出码: {result.returncode})")
            if result.stderr:
                print(f"     错误: {result.stderr[:200]}")
            results.append({'name': model['name'], 'status': '失败', 'time': elapsed})
            
    except subprocess.TimeoutExpired:
        print(f"     ⏱️ 超时 (>10分钟)")
        results.append({'name': model['name'], 'status': '超时', 'time': 600})
    except Exception as e:
        print(f"     ❌ 异常: {str(e)}")
        results.append({'name': model['name'], 'status': f'异常: {str(e)}', 'time': 0})

# 打印汇总
print("\n" + "=" * 60)
print("运行汇总")
print("=" * 60)
for r in results:
    status_icon = "✅" if r['status'] == '成功' else "❌"
    print(f"  {status_icon} {r['name']:<20} {r['status']:<10} ({r['time']:.1f}秒)")

total_time = sum(r['time'] for r in results)
print(f"\n  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

# 检查结果文件
print("\n" + "=" * 60)
print("生成的结果文件")
print("=" * 60)
if os.path.exists('results'):
    for f in sorted(os.listdir('results')):
        if f.endswith('.csv'):
            size = os.path.getsize(f'results/{f}')
            size_str = f"{size/1024:.0f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
            print(f"  📄 results/{f:<35} {size_str}")

print("\n" + "=" * 60)
print("全部完成！")
print("=" * 60)
