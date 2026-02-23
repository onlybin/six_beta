import requests
import json
import datetime
import sqlite3
import os

def init_db(db_path='lottery.db'):
    """初始化 SQLite 数据库表结构"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 创建开奖历史表。使用期号作为主键，开奖日期作为唯一索引(天然杜绝脏数据重复)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            period TEXT PRIMARY KEY,
            open_date TEXT UNIQUE,
            numbers TEXT,
            zodiacs TEXT,
            special INTEGER,
            special_zodiac TEXT,
            raw_time TEXT
        )
    ''')
    conn.commit()
    return conn

def fetch_lottery_data_api(db_path='lottery.db'):
    conn = init_db(db_path)
    cursor = conn.cursor()
    
    current_year = datetime.datetime.now().year
    years_to_fetch = [current_year, current_year - 1] 
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    for year in years_to_fetch:
        url = f"https://history.macaumarksix.com/history/macaujc2/y/{year}"
        print(f">>> 正在通过 API 拉取 {year} 年开奖数据(SQLite安全模式)...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() 
            data = response.json()
            
            if data.get('code') == 200 and data.get('result'):
                items = data.get('data', [])
                added_count = 0
                duplicate_count = 0
                
                for item in items:
                    open_time = item['openTime']
                    open_date = open_time.split(' ')[0]
                    period = item['expect']
                    
                    codes = [int(x) for x in item['openCode'].split(',')]
                    zodiacs = item['zodiac'].split(',')
                    
                    # SQLite 原生防呆去重：INSERT OR IGNORE
                    try:
                        cursor.execute('''
                            INSERT INTO history (period, open_date, numbers, zodiacs, special, special_zodiac, raw_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            period, 
                            open_date, 
                            json.dumps(codes[:6], ensure_ascii=False), 
                            json.dumps(zodiacs[:6], ensure_ascii=False), 
                            codes[6], 
                            zodiacs[6],
                            open_time
                        ))
                        added_count += 1
                    except sqlite3.IntegrityError:
                        # 触发了 UNIQUE 约束，说明数据已存在，直接跳过
                        duplicate_count += 1
                        
                conn.commit()
                print(f"    - {year}年接口：成功新增入库 {added_count} 条，拦截重复数据 {duplicate_count} 条")
            else:
                print(f"    - {year}年接口：数据获取失败: {data.get('message')}")
                
        except Exception as e:
            print(f"    - 请求 {year} 数据发生错误: {e}")
            
    # 统计数据库内的真实总数
    cursor.execute('SELECT COUNT(*) FROM history')
    total_records = cursor.fetchone()[0]
    conn.close()
        
    print(f"\n[成功] 数据库同步完毕！底层数据仓现绝对安全保留 {total_records} 条独立开奖记录。")

if __name__ == '__main__':
    # 注意：使用数据库后，不再需要暴力删除文件，直接追加执行即可
    fetch_lottery_data_api()