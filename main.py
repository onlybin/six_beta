import os
import subprocess
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# 文件路径配置
LOTTERY_DATA_FILE = 'lottery_complete.json'
ANALYSIS_RESULT_FILE = 'analysis_result.json'
PREDICTION_RESULT_FILE = 'prediction.json'
CHART_DATA_FILE = 'chart_data.json'
REPORT_FILE = 'lottery_analysis_report.md'

def run_script(script_name, *args):
    """基础运行函数：适用于不会报错的普通爬虫和数据分析组件"""
    cmd = [sys.executable, script_name] + list(args)
    print(f"\n>>> 正在运行: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if process.returncode != 0:
        print(f"错误: {script_name} 运行失败\n{process.stderr}")
        exit(1)
    print(process.stdout)
    return process.stdout

def run_predictor_with_fallback():
    """🌟 智能容灾降级机制：优先跑 Pro 版，报错则自动回退旧版"""
    print("\n>>> 🚀 尝试启动 [Pro 增强版] 双引擎推演 (predictor_pro.py)...")
    cmd_pro = [sys.executable, 'predictor_pro.py']
    process_pro = subprocess.run(cmd_pro, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    # 如果 Pro 版完美运行，直接输出并结束
    if process_pro.returncode == 0:
        print(process_pro.stdout)
        return
        
    # 如果 Pro 版报错了，拦截报错并触发降级方案
    print(f"⚠️ [Pro 版本运行异常] (系统已拦截):\n{process_pro.stderr}")
    print(">>> 🔄 触发自动降级保护：正在切换回 [基础稳定版] 单引擎推演 (predictor.py)...")
    
    cmd_base = [sys.executable, 'predictor.py']
    process_base = subprocess.run(cmd_base, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    # 检查基础版是否能挺住
    if process_base.returncode == 0:
        print(process_base.stdout)
    else:
        print(f"❌ [致命错误] 基础版也运行失败：\n{process_base.stderr}")
        exit(1)

def generate_report(latest_prediction, analysis_data):
    # 保留原有的 Markdown 报告生成逻辑
    print("\n>>> 正在组装全模态分析报告...")
    total_records = analysis_data.get('total_records', 0)
    
    special_rec_text = []
    top_specials = latest_prediction.get('recommendation', {}).get('special_numbers', [])
    for i, num in enumerate(top_specials):
        found = next((item for item in latest_prediction.get('top_scores', []) if item[0] == num), None)
        if found:
            score, zodiac = found[1], found[2]
            wuxing = found[3] if len(found)>3 else '?'
            color = found[4] if len(found)>4 else '?'
            if i == 0:
                special_rec_text.append(f"- **[首选] 第{i+1}名: {num:02d} ({zodiac}/{wuxing}/{color}波)** - 综合权重: **{score:.2f}** 🏆")
            else:
                special_rec_text.append(f"- 第{i+1}名: **{num:02d} ({zodiac}/{wuxing}/{color}波)** - 综合权重: {score:.2f}")
    special_text_block = '\n'.join(special_rec_text)

    normal_rec_text = []
    for num in latest_prediction.get('recommended_normal', []):
        found = next((item for item in latest_prediction.get('top_scores', []) if item[0] == num), None)
        normal_rec_text.append(f"- **{num:02d} ({found[2] if found else '?'})**")
    normal_text_block = '\n'.join(normal_rec_text)

    import datetime
    # 强制转换为东八区时间，适应云端 UTC 环境
    tz = datetime.timezone(datetime.timedelta(hours=8))
    report_time = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    
    attributes = latest_prediction.get('combo_attributes', {})

    report_content = f"""# 📊 AI 量化推演核心决策大屏

**最近更新时间:** {report_time} | **目标推演期数:** 第 {latest_prediction.get('next_period')} 期

> **[系统提示]** 基础算力平台已全面升级至 SQLite 关系型数据库底层，保障高并发分析安全。本期推演基于 {total_records} 期无损全量回溯。

---

### 🎯 2.1 特码预测 (高置信度矩阵)
*(注：列表依据孤立森林异常分、时序 MACD 动能及马尔可夫链转移概率综合降序排列)*
{special_text_block}

### 🎲 2.2 正码精选 (6个防守位)
{normal_text_block}

### ⚖️ 2.3 核心偏态指标
- **预测奇偶比:** {attributes.get('odd_even', '未知')}
- **预测大小比:** {attributes.get('big_small', '未知')}
- **7球预期和值:** {attributes.get('sum', '未知')}
"""
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
def main():
    # 清理旧缓存
    for f in [ANALYSIS_RESULT_FILE, PREDICTION_RESULT_FILE, CHART_DATA_FILE, REPORT_FILE]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    # 执行流水线
    run_script('fetcher.py')
    run_script('analyzer.py')
    
    # 🌟 调用带有容灾保护的推演引擎
    run_predictor_with_fallback()

    # 读取结果并生成大屏
    with open(PREDICTION_RESULT_FILE, 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)
    with open(ANALYSIS_RESULT_FILE, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    generate_report(prediction_data, analysis_data)
    print("\n=========================================")
    print("✅ 全自动化流水线执行完毕！请刷新网页大屏查看。")
    print("=========================================\n")

if __name__ == '__main__':
    main()