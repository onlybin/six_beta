from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import subprocess
import json
import os

# 初始化 FastAPI 应用
app = FastAPI(title="新澳门六合彩智能推演 API", version="2.0")

@app.get("/api/v1/prediction/latest")
async def get_latest_prediction():
    """
    获取最新一期的单注精选推演结果
    """
    # 1. 触发底层数据流 (复用你完善好的 main.py)
    # 这样每次外部调用接口，底层都会自动拉取最新数据、分析并推演
    try:
        print(">>> 收到 API 请求，正在运行核心推演流...")
        # 调用 python main.py
        subprocess.run(["python", "main.py"], check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "底层数据推演失败", "error_detail": e.stderr}
        )

    # 2. 读取推演生成的最终核心数据
    if not os.path.exists('prediction.json'):
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "尚未生成推演数据，请检查底层脚本"}
        )

    with open('prediction.json', 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)

    # 3. 包装成标准的前端/小程序 API 响应格式
    return {
        "code": 200,
        "message": "success",
        "data": {
            "target_period": prediction_data['next_period'],
            "based_on": prediction_data['based_on_period'],
            "recommendation": {
                # 严格输出唯一的一组数据
                "normal_numbers": prediction_data['recommended_normal'],
                "special_number": prediction_data['primary_special'],
                "special_zodiac": prediction_data['primary_special_zodiac']
            },
            "attributes": prediction_data['combo_attributes']
        }
    }

if __name__ == "__main__":
    # 启动 Web 服务器，监听本地 8000 端口
    print("🚀 API 服务器已启动！请在浏览器访问: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)