import json
import pandas as pd
import sqlite3
from collections import Counter

def get_records_from_db(db_path='lottery.db'):
    """从 SQLite 数据库提取结构化数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 按期号倒序提取 (最新期在最前)
    cursor.execute("SELECT period, raw_time, numbers, zodiacs, special, special_zodiac FROM history ORDER BY period DESC")
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            "period": row[0],
            "date": row[1],
            "numbers": json.loads(row[2]),
            "zodiacs": json.loads(row[3]),
            "special": row[4],
            "special_zodiac": row[5]
        })
    return records

def analyze_data(db_file='lottery.db', output_file='analysis_result.json', chart_file='chart_data.json'):
    print(">>> 正在进行深度数据清洗与 BI 数据集构建(从数据库拉取)...")
    records = get_records_from_db(db_file)

    if not records:
        raise ValueError("严重错误：数据库中没有任何开奖数据！请检查网络或接口是否异常。")

    df = pd.DataFrame(records)
    total_records = len(df)
    date_range = f"{df['date'].min().split()[0]} ~ {df['date'].max().split()[0]}"

    # 1. 计算遗漏值
    miss_values = {}
    for n in range(1, 50):
        miss = 0
        for record in records:
            if n in record['numbers'] or n == record['special']:
                break
            miss += 1
        miss_values[n] = miss

    # 2. 计算近50期冷热号
    recent_50 = records[:50]
    recent_nums = []
    for r in recent_50:
        recent_nums.extend(r['numbers'])
        recent_nums.append(r['special'])
    counter_50 = Counter(recent_nums)
    hot_cold = {n: counter_50.get(n, 0) for n in range(1, 50)}

    # 3. 计算生肖与波色分布 (特码)
    COLOR_MAP = {
        '红': [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46],
        '蓝': [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48],
        '绿': [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]
    }
    NUM_TO_COLOR = {n: c for c, nums in COLOR_MAP.items() for n in nums}
    
    zodiac_counts = dict(Counter([r['special_zodiac'] for r in records]))
    color_counts = dict(Counter([NUM_TO_COLOR.get(r['special'], '未知') for r in records]))

    analysis_result = {
        "total_records": total_records,
        "date_range": date_range,
        "miss_values": miss_values,
        "recent_50_hot": [k for k, v in counter_50.most_common(10)],
        "recent_50_cold": [k for k, v in counter_50.most_common()[-10:]]
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    chart_data = {
        "miss_values": miss_values,
        "hot_cold": hot_cold,
        "zodiac_counts": zodiac_counts,
        "color_counts": color_counts
    }
    with open(chart_file, 'w', encoding='utf-8') as f:
        json.dump(chart_data, f, ensure_ascii=False, indent=2)

    print(f"数据处理完毕！已生成模型源 {output_file} 和 BI源 {chart_file}")

if __name__ == '__main__':
    analyze_data()