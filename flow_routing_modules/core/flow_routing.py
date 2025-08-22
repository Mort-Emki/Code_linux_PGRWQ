"""
flow_routing.py - 高效无DataFrame流量计算核心

完全基于NumPy数组和内存映射的流量计算，避免任何DataFrame操作。
适合大规模深度学习项目的内存效率要求。
"""

import os
import numpy as np
import logging
import json
from typing import Dict

# 导入高效流量计算核心
from .efficient_flow_routing import create_efficient_flow_router


def flow_routing_calculation(df, 
                            iteration: int, 
                            model_func, 
                            river_info, 
                            v_f_TN: float = 35.0,
                            v_f_TP: float = 44.5,
                            attr_dict: dict = None, 
                            target_col: str = "TN",
                            **_):
    """
    高效流量计算主函数 - 完全无DataFrame模式
    
    参数:
        df: 包含二进制数据目录信息的特殊DataFrame
        iteration: 迭代次数
        model_func: 模型预测函数
        river_info: 河段信息DataFrame（仅用于构建拓扑）
        v_f_TN, v_f_TP: TN/TP的速度参数
        attr_dict: 属性字典
        target_col: 目标列("TN"或"TP")
        **kwargs: 其他参数（兼容性，忽略）
    
    返回:
        result_df: 计算结果DataFrame（仅用于接口兼容性）
    """
    logging.info(f"开始迭代 {iteration} 的高效流量计算 - {target_col}")
    
    # 验证输入数据格式
    if '_binary_mode' not in df.columns or not df['_binary_mode'].iloc[0]:
        raise ValueError("流量计算现在仅支持二进制数据输入。请先使用数据预处理脚本转换数据格式。")
    
    binary_data_dir = df['_binary_dir'].iloc[0]
    
    # 构建河网拓扑字典（从DataFrame转为字典，一次性转换）
    topology_dict = {}
    for _, row in river_info.iterrows():
        comid_str = str(row['COMID'])
        next_down_str = str(row['NextDownID'])
        if next_down_str not in ['0', 'nan', ''] and next_down_str != comid_str:
            topology_dict[comid_str] = next_down_str
    
    logging.info(f"构建了 {len(topology_dict)} 条拓扑连接")
    
    # 创建高效流量计算器
    flow_router = create_efficient_flow_router(
        binary_data_dir=binary_data_dir,
        topology_dict=topology_dict,
        attr_dict=attr_dict
    )
    
    # 选择速度参数
    v_f_param = v_f_TN if target_col == "TN" else v_f_TP
    
    # 设置临时输出目录
    temp_output_dir = f"/tmp/flow_results_iter_{iteration}_{target_col}"
    
    # 执行高效流量计算（完全无DataFrame）
    result_dir = flow_router.execute_flow_routing(
        target_col=target_col,
        model_predictor=model_func,
        v_f_param=v_f_param,
        output_binary_dir=temp_output_dir
    )
    
    # 转换结果为DataFrame格式（最小化，仅用于接口兼容性）
    result_df = _convert_binary_results_to_minimal_dataframe(
        result_dir, iteration, target_col, flow_router
    )
    
    logging.info(f"高效流量计算完成 - 内存占用极低")
    return result_df


def _convert_binary_results_to_minimal_dataframe(result_dir: str, 
                                               iteration: int, 
                                               target_col: str,
                                               flow_router):
    """
    将二进制结果转换为最小化DataFrame（仅用于接口兼容性）
    
    只包含非零结果，大幅减少内存占用
    """
    import pandas as pd
    
    # 读取结果元数据
    metadata_path = os.path.join(result_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    comid_list = metadata['comid_list']
    n_days = metadata['n_days']
    
    # 加载计算结果（内存映射模式，避免完整加载）
    y_up_data = np.load(os.path.join(result_dir, f'y_up_{target_col}.npy'), mmap_mode='r')
    y_n_data = np.load(os.path.join(result_dir, f'y_n_{target_col}.npy'), mmap_mode='r')
    
    # 获取日期数组
    dates = flow_router.dates_mmap[:]
    
    # 构建最小化结果（只保留有意义的数据点）
    result_rows = []
    
    for comid_idx, comid_str in enumerate(comid_list):
        y_up_vals = y_up_data[comid_idx]
        y_n_vals = y_n_data[comid_idx]
        
        # 只保留非零或有变化的数据点
        significant_indices = np.where(
            (np.abs(y_up_vals) > 1e-6) | (np.abs(y_n_vals) > 1e-6)
        )[0]
        
        if len(significant_indices) > 0:
            # 进一步采样，避免过多数据点
            if len(significant_indices) > 100:
                # 如果数据点过多，进行采样
                step = len(significant_indices) // 100
                significant_indices = significant_indices[::step]
            
            for day_idx in significant_indices:
                result_rows.append({
                    'COMID': int(comid_str),
                    'date': dates[day_idx],
                    f'y_up_{iteration}_{target_col}': float(y_up_vals[day_idx]),
                    f'y_n_{iteration}_{target_col}': float(y_n_vals[day_idx])
                })
    
    result_df = pd.DataFrame(result_rows)
    
    if len(result_df) > 0:
        # 按COMID和日期排序
        result_df = result_df.sort_values(['COMID', 'date']).reset_index(drop=True)
        
        logging.info(f"最小化结果转换完成：{len(result_df)} 行有效数据（原始: {len(comid_list) * n_days}）")
        logging.info(f"数据压缩比：{len(result_df) / (len(comid_list) * n_days) * 100:.1f}%")
    else:
        logging.warning("未发现有效的计算结果")
    
    return result_df


def get_flow_calculation_memory_usage():
    """
    获取当前流量计算的内存使用情况
    
    返回:
        内存使用统计字典
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
        'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
        'percent': process.memory_percent(),       # 内存使用百分比
        'mode': 'efficient_numpy_mmap'             # 模式标识
    }


def validate_binary_data_format(df) -> bool:
    """
    验证输入数据是否为正确的二进制格式
    
    参数:
        df: 输入数据框
        
    返回:
        bool: 是否为有效的二进制格式
    """
    try:
        if '_binary_mode' not in df.columns:
            return False
            
        if not df['_binary_mode'].iloc[0]:
            return False
            
        binary_dir = df['_binary_dir'].iloc[0]
        
        # 检查必要文件是否存在
        required_files = ['metadata.json', 'data.npy', 'dates.npy']
        
        for file_name in required_files:
            file_path = os.path.join(binary_dir, file_name)
            if not os.path.exists(file_path):
                logging.error(f"缺少必要文件: {file_path}")
                return False
        
        # 验证元数据格式
        metadata_path = os.path.join(binary_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        required_fields = ['n_comids', 'n_days', 'feature_columns', 'comid_list']
        for field in required_fields:
            if field not in metadata:
                logging.error(f"元数据缺少字段: {field}")
                return False
        
        logging.info("二进制数据格式验证通过")
        return True
        
    except Exception as e:
        logging.error(f"二进制数据格式验证失败: {e}")
        return False


# 简化的工厂函数
def create_flow_calculator(binary_data_dir: str, topology_dict: Dict[str, str], attr_dict: dict = None):
    """
    创建流量计算器的简化工厂函数
    
    参数:
        binary_data_dir: 二进制数据目录
        topology_dict: 拓扑字典
        attr_dict: 属性字典
        
    返回:
        配置好的流量计算器
    """
    return create_efficient_flow_router(
        binary_data_dir=binary_data_dir,
        topology_dict=topology_dict,
        attr_dict=attr_dict
    )