"""
utils.py - 辅助函数模块

提供各种实用函数，如结果处理、路径管理、批量函数创建等。
简化主逻辑中的重复代码，提高可维护性。
"""

import os
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime

from ..gpu_memory_utils import TimingAndMemoryContext
from ...logging_utils import ensure_dir_exists


def check_existing_flow_routing_results(
    iteration: int, 
    model_version: str, 
    flow_results_dir: str
) -> Tuple[bool, str]:
    """
    检查是否已存在特定迭代和模型版本的汇流计算结果文件
    
    参数:
        iteration: 迭代次数
        model_version: 模型版本号
        flow_results_dir: 汇流结果保存目录
        
    返回:
        (exists, file_path): 元组，包含是否存在的布尔值和CSV文件路径
        
    注意:
        优先检查二进制格式，如果不存在则检查CSV格式
    """
    # 检查二进制格式
    binary_dir = os.path.join(flow_results_dir, f"flow_routing_iteration_{iteration}_{model_version}_binary")
    binary_metadata = os.path.join(binary_dir, 'metadata.json')
    
    if os.path.exists(binary_metadata):
        # 二进制格式存在，返回对应的CSV路径（保持兼容性）
        csv_path = os.path.join(flow_results_dir, f"flow_routing_iteration_{iteration}_{model_version}.csv")
        return True, csv_path
    
    # 检查CSV格式
    csv_path = os.path.join(flow_results_dir, f"flow_routing_iteration_{iteration}_{model_version}.csv")
    csv_exists = os.path.isfile(csv_path)
    return csv_exists, csv_path


def create_predictor(data_handler, model_manager, all_target_cols, target_col):
    """创建预测器实例"""
    from .predictor import CatchmentPredictor
    return CatchmentPredictor(data_handler, model_manager, all_target_cols, target_col)

def save_flow_results(df_flow, iteration, model_version, output_dir):
    """
    保存汇流计算结果（CSV + 二进制格式）
    
    参数:
        df_flow: 汇流计算结果DataFrame
        iteration: 迭代次数
        model_version: 模型版本号
        output_dir: 输出目录
        
    返回:
        binary_dir: 二进制数据目录路径
    """
    # 确保目录存在
    ensure_dir_exists(output_dir)
    
    # 保存CSV格式（保持兼容性）
    csv_path = os.path.join(output_dir, f"flow_routing_iteration_{iteration}_{model_version}.csv")
    df_flow.to_csv(csv_path, index=False)
    logging.info(f"迭代 {iteration} 汇流计算结果（CSV）已保存至 {csv_path}")
    
    # 转换并保存二进制格式
    binary_dir = os.path.join(output_dir, f"flow_routing_iteration_{iteration}_{model_version}_binary")
    
    try:
        # 导入二进制转换工具
        import sys
        import subprocess
        
        # 调用CSV到二进制转换脚本
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts', 'csv_to_binary_converter.py')
        
        if os.path.exists(script_path):
            cmd = [sys.executable, script_path, '--input', csv_path, '--output', binary_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"迭代 {iteration} 汇流计算结果（二进制）已保存至 {binary_dir}")
            else:
                logging.warning(f"二进制转换失败: {result.stderr}")
                binary_dir = None
        else:
            logging.warning(f"找不到转换脚本: {script_path}")
            binary_dir = None
            
    except Exception as e:
        logging.warning(f"转换为二进制格式时出错: {e}")
        binary_dir = None
    
    return binary_dir


def time_function(func):
    """
    函数执行时间装饰器
    
    参数:
        func: 要计时的函数
        
    返回:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"函数 {func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result
    return wrapper


def split_train_val_data(
    X_ts: np.ndarray, 
    Y: np.ndarray, 
    COMIDs: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple:
    """
    将数据划分为训练集和验证集
    
    参数:
        X_ts: 时间序列数据
        Y: 目标变量
        COMIDs: COMID数组
        train_ratio: 训练集比例
        
    返回:
        (X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val): 划分后的数据
    """
    with TimingAndMemoryContext("训练/验证集划分"):
        N = len(X_ts)
        indices = np.random.permutation(N)
        train_size = int(N * train_ratio)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

        X_ts_train = X_ts[train_indices]
        comid_arr_train = COMIDs[train_indices]
        Y_train = Y[train_indices]

        X_ts_val = X_ts[valid_indices]
        comid_arr_val = COMIDs[valid_indices]
        Y_val = Y[valid_indices]

    return X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val