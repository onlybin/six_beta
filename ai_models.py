# ai_models.py
# 独立的 AI 算法模块：负责 RF、XGBoost 与 LSTM 三大引擎的训练及融合预测

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

# 引入 TensorFlow/Keras 深度学习相关库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import os
# 禁用 TensorFlow 烦人的底层警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_anomaly_scores(X_train, X_predict):
    """引擎0：孤立森林（计算异常分，用于过滤极端偏态）"""
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)
    train_scores = iso_forest.decision_function(X_train)
    predict_scores = iso_forest.decision_function(X_predict)
    return train_scores, predict_scores

def get_ensemble_probabilities(X_train, y_train, X_predict):
    """引擎1 & 2 & 3：训练 RF, XGBoost, LSTM，并返回异构融合后的概率"""
    
    # --------------------------------------------------
    # 引擎 1: Random Forest
    # --------------------------------------------------
    print(">>> [AI 核心模块] 启动引擎 1: 随机森林 (AutoML 寻优)...")
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_param = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 5]
    }
    rf_search = RandomizedSearchCV(base_rf, param_distributions=rf_param, n_iter=3, cv=3, scoring='roc_auc', random_state=42, n_jobs=1)
    rf_search.fit(X_train, y_train)
    rf_prob = rf_search.best_estimator_.predict_proba(X_predict)[:, 1]

    # --------------------------------------------------
    # 引擎 2: XGBoost
    # --------------------------------------------------
    print(">>> [AI 核心模块] 启动引擎 2: XGBoost 梯度提升树 (AutoML 寻优)...")
    base_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_param = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    xgb_search = RandomizedSearchCV(base_xgb, param_distributions=xgb_param, n_iter=3, cv=3, scoring='roc_auc', random_state=42, n_jobs=1)
    xgb_search.fit(X_train, y_train)
    xgb_prob = xgb_search.best_estimator_.predict_proba(X_predict)[:, 1]

    # --------------------------------------------------
    # 引擎 3: LSTM 深度神经网络
    # --------------------------------------------------
    print(">>> [AI 核心模块] 启动引擎 3: LSTM 长短期记忆神经网络...")
    
    # 降维重组：LSTM 要求输入 3D 张量 (样本数, 时间步, 特征数)
    X_train_np = np.array(X_train)
    X_predict_np = np.array(X_predict)
    y_train_np = np.array(y_train)

    X_train_lstm = np.reshape(X_train_np, (X_train_np.shape[0], 1, X_train_np.shape[1]))
    X_predict_lstm = np.reshape(X_predict_np, (X_predict_np.shape[0], 1, X_predict_np.shape[1]))

    # 构建 LSTM 网络拓扑
    lstm_model = Sequential()
    lstm_model.add(LSTM(32, input_shape=(1, X_train_np.shape[1]), activation='relu'))
    lstm_model.add(Dropout(0.2)) # 防止过拟合
    lstm_model.add(Dense(16, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid')) # 输出概率

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 拟合网络 (配置早停机制，避免云端计算超时)
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train_np, epochs=15, batch_size=32, verbose=0, callbacks=[early_stop])
    
    lstm_prob = lstm_model.predict(X_predict_lstm, verbose=0).flatten()

    # --------------------------------------------------
    # 三级异构网络 Stacking 融合
    # --------------------------------------------------
    print(">>> [AI 核心模块] 计算完毕，正在执行概率融合 (RF 35% + XGB 35% + LSTM 30%)...")
    
    # 权重解释：树模型处理离散静态特征更稳定占主导，LSTM 补充捕捉非线性动态序列占 30%
    ensemble_probabilities = (rf_prob * 0.45) + (xgb_prob * 0.45) + (lstm_prob * 0.10)
    
    return ensemble_probabilities