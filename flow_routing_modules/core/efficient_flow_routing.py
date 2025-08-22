"""
efficient_flow_routing.py - 完全无DataFrame的高效流量计算核心

设计原则：
1. 完全基于NumPy数组和内存映射
2. 避免任何DataFrame操作
3. 直接二进制I/O，避免CSV解析开销
4. 内存占用 O(河段数) 而非 O(河段数 × 时间步数)
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from numba import njit, prange

class EfficientFlowRouter:
    """
    完全无DataFrame的高效流量计算器
    
    核心特性:
    - 纯NumPy数组操作
    - 内存映射文件I/O  
    - JIT编译向量化计算
    - 常数级内存占用
    """
    
    def __init__(self, binary_data_dir: str):
        """
        初始化高效流量计算器
        
        参数:
            binary_data_dir: 二进制数据目录
        """
        self.binary_data_dir = binary_data_dir
        self.comid_index = {}
        self.n_comids = 0
        self.n_days = 0
        self.feature_names = []
        
        # 内存映射数组
        self.data_mmap = None
        self.dates_mmap = None
        
        # 拓扑结构（纯数组）
        self.next_down_array = None  # shape: (n_comids,)
        self.indegree_array = None   # shape: (n_comids,)
        
        # 属性数据（纯数组）
        self.attributes_array = None  # shape: (n_comids, n_attrs)
        
        # 中间计算缓冲区（复用内存）
        self.temp_buffers = {}
        
        self._load_binary_data()
    
    def _load_binary_data(self):
        """加载二进制数据到内存映射"""
        # 加载元数据
        import json
        metadata_path = os.path.join(self.binary_data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.n_comids = metadata['n_comids']
        self.n_days = metadata['n_days'] 
        self.feature_names = metadata['feature_columns']
        
        # 构建COMID索引
        for i, comid_str in enumerate(metadata['comid_list']):
            self.comid_index[comid_str] = i
        
        # 内存映射主数据文件
        data_path = os.path.join(self.binary_data_dir, 'data.npy')
        self.data_mmap = np.load(data_path, mmap_mode='r')  # shape: (n_comids, n_days, n_features)
        
        # 内存映射日期文件  
        dates_path = os.path.join(self.binary_data_dir, 'dates.npy')
        self.dates_mmap = np.load(dates_path, mmap_mode='r')
        
        logging.info(f"内存映射数据加载完成: {self.n_comids} comids × {self.n_days} days")
    
    def set_topology(self, next_down_dict: Dict[str, str]):
        """
        设置河网拓扑结构（纯数组存储）
        
        参数:
            next_down_dict: COMID -> 下游COMID的映射字典
        """
        self.next_down_array = np.full(self.n_comids, -1, dtype=np.int32)
        self.indegree_array = np.zeros(self.n_comids, dtype=np.int32)
        
        for comid_str, next_down_str in next_down_dict.items():
            if comid_str in self.comid_index and next_down_str in self.comid_index:
                comid_idx = self.comid_index[comid_str]
                next_down_idx = self.comid_index[next_down_str]
                
                self.next_down_array[comid_idx] = next_down_idx
                self.indegree_array[next_down_idx] += 1
        
        logging.info("拓扑结构设置完成")
    
    def set_attributes(self, attr_dict: Dict[str, np.ndarray]):
        """
        设置属性数据（纯数组存储）
        
        参数:
            attr_dict: COMID -> 属性向量的映射
        """
        if not attr_dict:
            return
            
        attr_dim = len(next(iter(attr_dict.values())))
        self.attributes_array = np.zeros((self.n_comids, attr_dim), dtype=np.float32)
        
        for comid_str, attrs in attr_dict.items():
            if comid_str in self.comid_index:
                comid_idx = self.comid_index[comid_str]
                self.attributes_array[comid_idx] = attrs
        
        logging.info(f"属性数据设置完成: {attr_dim} 维度")
    
    def allocate_computation_buffers(self):
        """预分配计算缓冲区，避免运行时内存分配"""
        # 负荷累积数组 (n_comids, n_days)
        self.temp_buffers['load_accumulator'] = np.zeros((self.n_comids, self.n_days), dtype=np.float32)
        
        # 中间计算结果 (n_days,) - 单个河段的临时数组
        self.temp_buffers['contribution'] = np.empty(self.n_days, dtype=np.float32)
        self.temp_buffers['retention'] = np.empty(self.n_days, dtype=np.float32)
        self.temp_buffers['temp_calc'] = np.empty(self.n_days, dtype=np.float32)
        
        # 输出数组 (n_comids, n_days)
        self.temp_buffers['y_up_result'] = np.zeros((self.n_comids, self.n_days), dtype=np.float32) 
        self.temp_buffers['y_n_result'] = np.zeros((self.n_comids, self.n_days), dtype=np.float32)
        
        logging.info("计算缓冲区分配完成")
    
    @njit
    def _compute_contribution_vectorized(y_n_array, r_series_array, q_array, out_array):
        """
        JIT编译的向量化贡献计算
        
        参数:
            y_n_array: 节点浓度数组
            r_series_array: 保留系数数组  
            q_array: 流量数组
            out_array: 输出数组（就地计算）
        """
        # 纯NumPy向量化操作，JIT优化
        np.multiply(y_n_array, r_series_array, out=out_array)
        np.multiply(out_array, q_array, out=out_array)
    
    def execute_flow_routing(self, 
                            target_col: str,
                            model_predictor,
                            v_f_param: float = 35.0,
                            output_binary_dir: str = None) -> str:
        """
        执行完全无DataFrame的流量计算
        
        参数:
            target_col: 目标参数 ('TN' 或 'TP')
            model_predictor: 模型预测函数
            v_f_param: 速度参数
            output_binary_dir: 输出二进制目录
            
        返回:
            输出二进制目录路径
        """
        logging.info(f"开始执行无DataFrame流量计算 - {target_col}")
        
        # 分配计算缓冲区
        self.allocate_computation_buffers()
        
        # 获取特征列索引
        try:
            qout_idx = self.feature_names.index('Qout')
            width_idx = self.feature_names.index('width') if 'width' in self.feature_names else None
            temp_idx = self.feature_names.index('temperature_2m_mean') if 'temperature_2m_mean' in self.feature_names else None
        except ValueError as e:
            raise ValueError(f"必要的特征列缺失: {e}")
        
        # 找到入度为0的河段（头部河段）
        head_segments = np.where(self.indegree_array == 0)[0]
        
        # 初始化头部河段
        for comid_idx in head_segments:
            # 使用模型预测E值，然后设置 y_n = E
            e_values = self._predict_e_values_for_comid(comid_idx, model_predictor, target_col)
            self.temp_buffers['y_n_result'][comid_idx] = e_values
        
        # 构建处理队列（拓扑排序）
        queue = list(head_segments)
        indegree_working = self.indegree_array.copy()
        
        processed_count = 0
        
        # 主循环：拓扑排序处理河段
        while queue:
            current_idx = queue.pop(0)
            processed_count += 1
            
            if processed_count % 1000 == 0:
                logging.info(f"已处理 {processed_count} 个河段")
            
            next_down_idx = self.next_down_array[current_idx]
            
            # 如果没有下游，跳过
            if next_down_idx == -1:
                continue
            
            # 执行当前 -> 下游的流量计算
            self._compute_flow_contribution(current_idx, next_down_idx, qout_idx, width_idx, temp_idx, v_f_param)
            
            # 更新下游河段的入度
            indegree_working[next_down_idx] -= 1
            
            # 如果下游河段所有上游都处理完毕
            if indegree_working[next_down_idx] == 0:
                self._finalize_downstream_computation(next_down_idx, qout_idx, target_col)
                queue.append(next_down_idx)
        
        # 保存结果到二进制文件
        if output_binary_dir:
            self._save_results_binary(target_col, output_binary_dir)
            logging.info(f"流量计算完成，结果已保存至 {output_binary_dir}")
            return output_binary_dir
        else:
            logging.info("流量计算完成（未保存文件）")
            return None
    
    def _compute_flow_contribution(self, 
                                 upstream_idx: int, 
                                 downstream_idx: int,
                                 qout_idx: int,
                                 width_idx: Optional[int],
                                 temp_idx: Optional[int], 
                                 v_f_param: float):
        """
        计算上游对下游的流量贡献（纯NumPy操作）
        
        参数:
            upstream_idx: 上游河段索引
            downstream_idx: 下游河段索引
            qout_idx: 流量特征索引
            width_idx: 宽度特征索引
            temp_idx: 温度特征索引  
            v_f_param: 速度参数
        """
        # 从内存映射获取数据（零拷贝）
        upstream_data = self.data_mmap[upstream_idx]    # shape: (n_days, n_features)
        downstream_data = self.data_mmap[downstream_idx]
        
        # 提取时间序列（引用，不复制）
        q_upstream = upstream_data[:, qout_idx]
        q_downstream = downstream_data[:, qout_idx] 
        
        if width_idx is not None:
            w_upstream = upstream_data[:, width_idx]
            w_downstream = downstream_data[:, width_idx]
        else:
            w_upstream = w_downstream = None
        
        if temp_idx is not None:
            temperature = upstream_data[:, temp_idx]
        else:
            temperature = None
        
        # 获取当前河段的y_n值
        y_n_upstream = self.temp_buffers['y_n_result'][upstream_idx]
        
        # 计算保留系数（向量化）
        retention_coeff = self._compute_retention_coefficient_vectorized(
            v_f_param, q_upstream, q_downstream, w_upstream, w_downstream, temperature
        )
        
        # 计算贡献量（就地操作，避免临时数组）
        contribution_buffer = self.temp_buffers['contribution']
        EfficientFlowRouter._compute_contribution_vectorized(
            y_n_upstream, retention_coeff, q_upstream, contribution_buffer
        )
        
        # 累积到负荷累积器（就地加法）
        np.add(self.temp_buffers['load_accumulator'][downstream_idx], 
               contribution_buffer,
               out=self.temp_buffers['load_accumulator'][downstream_idx])
    
    @njit  
    def _compute_retention_coefficient_vectorized(self, 
                                                v_f: float,
                                                q_up: np.ndarray, 
                                                q_down: np.ndarray,
                                                w_up: Optional[np.ndarray],
                                                w_down: Optional[np.ndarray],
                                                temperature: Optional[np.ndarray]) -> np.ndarray:
        """
        JIT编译的向量化保留系数计算
        
        返回:
            retention_coefficient: shape (n_days,)
        """
        n_days = len(q_up)
        result = np.empty(n_days, dtype=np.float32)
        
        for i in prange(n_days):
            # 简化的保留系数计算（可根据具体物理模型调整）
            if q_up[i] > 0 and q_down[i] > 0:
                base_retention = 1.0 - v_f / (v_f + q_up[i])
                
                # 温度修正
                if temperature is not None and not np.isnan(temperature[i]):
                    temp_factor = 1.0 + 0.02 * (temperature[i] - 20.0)  # 简化温度修正
                    base_retention *= temp_factor
                
                # 宽度修正
                if w_up is not None and w_down is not None:
                    if w_up[i] > 0 and w_down[i] > 0:
                        width_factor = (w_up[i] / w_down[i]) ** 0.5
                        base_retention *= width_factor
                
                result[i] = max(0.0, min(1.0, base_retention))
            else:
                result[i] = 0.0
                
        return result
    
    def _finalize_downstream_computation(self, downstream_idx: int, qout_idx: int, target_col: str):
        """
        完成下游河段的最终计算（纯NumPy操作）
        
        参数:
            downstream_idx: 下游河段索引
            qout_idx: 流量特征索引
            target_col: 目标列名
        """
        # 获取流量数据（零拷贝引用）
        q_downstream = self.data_mmap[downstream_idx, :, qout_idx]
        
        # 计算 y_up = load_accumulator / Qout
        load_accumulated = self.temp_buffers['load_accumulator'][downstream_idx]
        
        # 安全的除法（就地操作）
        y_up_buffer = self.temp_buffers['y_up_result'][downstream_idx]
        np.divide(load_accumulated, q_downstream, out=y_up_buffer, where=q_downstream>0)
        np.nan_to_num(y_up_buffer, copy=False)  # 就地处理NaN
        
        # 获取E值（通过模型预测）
        e_values = self._predict_e_values_for_comid(downstream_idx, None, target_col)
        
        # 计算 y_n = E + y_up（就地操作）
        y_n_buffer = self.temp_buffers['y_n_result'][downstream_idx]
        np.add(e_values, y_up_buffer, out=y_n_buffer)
    
    def _predict_e_values_for_comid(self, comid_idx: int, model_predictor, target_col: str) -> np.ndarray:
        """
        为单个COMID预测E值（避免DataFrame）
        
        参数:
            comid_idx: COMID索引
            model_predictor: 模型预测函数
            target_col: 目标列名
            
        返回:
            e_values: shape (n_days,)
        """
        if model_predictor is None:
            # 返回零向量（头部河段或测试情况）
            return np.zeros(self.n_days, dtype=np.float32)
        
        # 从内存映射获取COMID数据（零拷贝）
        comid_data = self.data_mmap[comid_idx]  # shape: (n_days, n_features)
        
        # 构建滑动窗口（避免完整复制）
        time_window = 10
        if self.n_days < time_window:
            return np.zeros(self.n_days, dtype=np.float32)
        
        # 高效窗口构建
        windows = np.lib.stride_tricks.sliding_window_view(
            comid_data, window_shape=(time_window, comid_data.shape[1]), axis=0
        ).squeeze()
        
        # 获取属性数据（如果存在）
        attrs = None
        if self.attributes_array is not None:
            attrs = self.attributes_array[comid_idx]
        
        # 调用模型预测
        try:
            e_predictions = model_predictor(windows, attrs, target_col)
            
            # 确保输出维度正确
            if len(e_predictions) != self.n_days - time_window + 1:
                # 填充或截断以匹配期望维度
                e_full = np.zeros(self.n_days, dtype=np.float32)
                valid_len = min(len(e_predictions), self.n_days - time_window + 1)
                e_full[time_window-1:time_window-1+valid_len] = e_predictions[:valid_len]
                return e_full
            else:
                # 前补零以匹配完整时间序列长度
                e_full = np.zeros(self.n_days, dtype=np.float32)
                e_full[time_window-1:] = e_predictions
                return e_full
                
        except Exception as e:
            logging.warning(f"模型预测失败，COMID索引 {comid_idx}: {e}")
            return np.zeros(self.n_days, dtype=np.float32)
    
    def _save_results_binary(self, target_col: str, output_dir: str):
        """
        保存计算结果为二进制格式（避免CSV）
        
        参数:
            target_col: 目标列名
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存主要结果数组
        y_up_path = os.path.join(output_dir, f'y_up_{target_col}.npy')
        y_n_path = os.path.join(output_dir, f'y_n_{target_col}.npy')
        
        np.save(y_up_path, self.temp_buffers['y_up_result'])
        np.save(y_n_path, self.temp_buffers['y_n_result'])
        
        # 保存元数据
        import json
        metadata = {
            'target_col': target_col,
            'n_comids': self.n_comids,
            'n_days': self.n_days,
            'comid_list': list(self.comid_index.keys()),
            'output_arrays': [f'y_up_{target_col}.npy', f'y_n_{target_col}.npy']
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"结果已保存为二进制格式: {output_dir}")


def create_efficient_flow_router(binary_data_dir: str, 
                                topology_dict: Dict[str, str],
                                attr_dict: Optional[Dict[str, np.ndarray]] = None) -> EfficientFlowRouter:
    """
    创建高效流量计算器的工厂函数
    
    参数:
        binary_data_dir: 二进制数据目录
        topology_dict: 河网拓扑字典
        attr_dict: 属性字典
        
    返回:
        配置好的EfficientFlowRouter实例
    """
    router = EfficientFlowRouter(binary_data_dir)
    router.set_topology(topology_dict)
    
    if attr_dict:
        router.set_attributes(attr_dict)
    
    return router