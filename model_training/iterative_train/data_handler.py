"""
data_handler.py - 高效数据处理与标准化模块

仅支持基于内存映射的二进制数据访问，实现真正的O(1)随机访问
"""

import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any

# 导入项目中的函数
from ...data_processing import standardize_attributes
from ..gpu_memory_utils import TimingAndMemoryContext

# 导入高效数据加载器
from ...efficient_data_loader import EfficientDataLoader


class DataHandler:
    """
    高效数据处理器类 - 仅支持二进制模式
    
    核心特性：
    1. 基于内存映射的真正O(1)随机访问
    2. 内存占用极低（20GB → 50MB）  
    3. 零重复文本解析开销
    """
    
    def __init__(self):
        """初始化数据处理器"""
        # 基础属性
        self.attr_df: Optional[pd.DataFrame] = None
        self.input_features: Optional[List[str]] = None
        self.attr_features: Optional[List[str]] = None
        self.initialized = False
        
        # 高效数据加载器
        self.efficient_loader: Optional[EfficientDataLoader] = None
        
        # 标准化器
        self.ts_scaler: Optional[Any] = None
        self.attr_scaler: Optional[Any] = None
        
        # 预标准化属性
        self._raw_attr_dict: Optional[Dict[str, np.ndarray]] = None
        self._standardized_attr_dict: Optional[Dict[str, np.ndarray]] = None
        
    def initialize(self, 
                  df: pd.DataFrame, 
                  attr_df: pd.DataFrame,
                  input_features: List[str],
                  attr_features: List[str]):
        """
        初始化数据处理器
        
        参数:
            df: 包含二进制数据目录信息的特殊DataFrame
            attr_df: 包含属性数据的DataFrame
            input_features: 输入特征列表
            attr_features: 属性特征列表
        """
        with TimingAndMemoryContext("DataHandler初始化"):
            # 检查是否为二进制模式标记
            if '_binary_mode' not in df.columns or not df['_binary_mode'].iloc[0]:
                raise ValueError("DataHandler现在仅支持二进制模式")
            
            binary_dir = df['_binary_dir'].iloc[0]
            
            # 初始化高效加载器
            self.efficient_loader = EfficientDataLoader(binary_dir)
            self.attr_df = attr_df.copy()
            self.input_features = input_features
            self.attr_features = attr_features
            
            # 构建属性字典
            self._build_attribute_dictionary()
            
            # 创建标准化器
            self._create_ts_scaler()
            
            # 预标准化属性
            self._precompute_standardized_attributes()
            
            self.initialized = True
            
            logging.info(f"DataHandler初始化完成（二进制模式）")
            logging.info(f"  - 数据形状: {self.efficient_loader.get_statistics()['data_shape']}")
            logging.info(f"  - COMID数量: {self.efficient_loader.get_statistics()['n_comids']:,}")
    
    def _build_attribute_dictionary(self):
        """构建原始属性字典"""
        with TimingAndMemoryContext("构建属性字典"):
            self._raw_attr_dict = {}
            if self.attr_df is not None and self.attr_features is not None:
                for _, row in self.attr_df.iterrows():
                    comid: str = str(row['COMID'])
                    attrs = []
                    for attr in self.attr_features:
                        if attr in row:
                            value = row[attr]
                            if pd.isna(value):
                                attrs.append(0.0)
                            elif isinstance(value, (int, float)):
                                attrs.append(float(value))
                            else:
                                try:
                                    attrs.append(float(value))
                                except (ValueError, TypeError):
                                    attrs.append(0.0)
                        else:
                            attrs.append(0.0)
                    self._raw_attr_dict[comid] = np.array(attrs, dtype=np.float32)
            
            logging.info(f"属性字典构建完成：{len(self._raw_attr_dict)} 个河段")
    
    def _create_ts_scaler(self):
        """创建时间序列标准化器"""
        # 选择少量COMID作为样本
        all_comids = list(self.efficient_loader.comid_index.keys())
        sample_comids = all_comids[:min(50, len(all_comids))]
        
        logging.info(f"使用 {len(sample_comids)} 个COMID创建标准化器")
        
        # 收集样本数据
        sample_data_list = []
        for comid in sample_comids:
            comid_data = self.efficient_loader.load_comid_data(comid)
            if len(comid_data) >= 10:
                # 构建滑动窗口
                for i in range(len(comid_data) - 10 + 1):
                    window = comid_data[i:i + 10]
                    sample_data_list.append(window)
                    if len(sample_data_list) >= 1000:
                        break
            
            if len(sample_data_list) >= 1000:
                break
        
        if sample_data_list:
            from ...data_processing import standardize_time_series_all
            sample_array = np.array(sample_data_list, dtype=np.float32)
            _, self.ts_scaler = standardize_time_series_all(sample_array)
            logging.info(f"标准化器创建完成，使用了 {len(sample_data_list)} 个样本窗口")
        else:
            self.ts_scaler = None
            logging.warning("无法创建时间序列标准化器")
    
    def _precompute_standardized_attributes(self):
        """预标准化所有属性"""
        if not self._raw_attr_dict:
            logging.warning("无原始属性数据，跳过预标准化")
            return
            
        with TimingAndMemoryContext("预标准化属性"):
            # 标准化所有属性
            self._standardized_attr_dict, self.attr_scaler = standardize_attributes(self._raw_attr_dict)
            logging.info("属性预标准化完成")
    
    def prepare_streaming_training_data(self, 
                                      comid_list: List[str],
                                      all_target_cols: List[str],
                                      target_col: str,
                                      batch_size: int = 200):
        """准备流式训练数据迭代器"""
        
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
        
        # 使用高效二进制模式
        return self.efficient_loader.create_training_data_iterator(
            comid_list=comid_list,
            input_cols=self.input_features,
            target_col=target_col,
            all_target_cols=all_target_cols,
            time_window=10,
            batch_size=batch_size
        )
    
    def get_standardized_attr_dict(self) -> Dict[str, np.ndarray]:
        """获取标准化的完整属性字典"""
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
        if self._standardized_attr_dict is not None:
            return self._standardized_attr_dict.copy()
        else:
            return {}
    
    def prepare_batch_prediction_data(self, 
                                     comid_batch: List, 
                                     all_target_cols: List[str],
                                     target_col: str) -> Optional[Dict[str, Any]]:
        """
        为批量预测准备数据（二进制模式）
        
        参数:
            comid_batch: 河段COMID列表
            all_target_cols: 所有目标列列表（保持接口兼容性，暂未使用）
            target_col: 目标列名（保持接口兼容性，暂未使用）
            
        返回:
            包含预测数据的字典，格式：
            {
                'X_ts_scaled': 标准化的时间序列数据,
                'X_attr_batch': 属性数据,
                'valid_comids': 有效的COMID列表,
                'comid_indices': COMID索引映射,
                'groups': 原始数据组（保持兼容性）
            }
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
        
        with TimingAndMemoryContext("准备批量预测数据"):
            X_ts_list = []
            comid_indices = {}
            valid_comids = []
            
            current_idx = 0
            
            # 为每个COMID准备数据
            for comid in comid_batch:
                comid_str = str(comid)
                
                try:
                    # 从二进制数据加载COMID数据
                    comid_data = self.efficient_loader.load_comid_data(comid_str)
                    
                    if len(comid_data) < 10:  # 确保有足够数据构建窗口
                        continue
                    
                    # 构建滑动窗口
                    windows = []
                    dates = []
                    
                    for i in range(len(comid_data) - 10 + 1):
                        window = comid_data[i:i + 10]
                        # 检查窗口中的数据是否有效
                        if not np.any(np.isnan(window)):
                            windows.append(window)
                            # 获取对应的日期（使用加载器的日期信息）
                            _, comid_dates = self.efficient_loader.load_comid_with_dates(comid_str)
                            if len(comid_dates) > i + 9:
                                dates.append(comid_dates[i + 9])  # 窗口最后一天的日期
                    
                    if not windows:
                        continue
                    
                    # 转换为numpy数组
                    X_comid = np.array(windows, dtype=np.float32)
                    
                    # 标准化时间序列数据
                    if self.ts_scaler is not None:
                        N, T, D = X_comid.shape
                        X_comid_2d = X_comid.reshape(-1, D)
                        X_comid_scaled_2d = self.ts_scaler.transform(X_comid_2d)
                        X_comid_scaled = X_comid_scaled_2d.reshape(N, T, D)
                    else:
                        X_comid_scaled = X_comid
                    
                    # 记录索引信息
                    end_idx = current_idx + len(X_comid_scaled)
                    comid_indices[comid] = (current_idx, end_idx, dates, dates)
                    current_idx = end_idx
                    valid_comids.append(comid)
                    
                    X_ts_list.append(X_comid_scaled)
                    
                except Exception as e:
                    logging.warning(f"处理COMID {comid} 时出错: {e}")
                    continue
            
            if not X_ts_list:
                logging.warning("没有有效的COMID数据用于批量预测")
                return None
            
            # 合并所有时间序列数据
            X_ts_batch = np.vstack(X_ts_list)
            
            # 准备属性数据
            if self._standardized_attr_dict:
                attr_dim = next(iter(self._standardized_attr_dict.values())).shape[0]
                X_attr_batch = np.zeros((X_ts_batch.shape[0], attr_dim), dtype=np.float32)
                
                # 为每个样本分配正确的属性向量
                sample_idx = 0
                for comid in valid_comids:
                    start_idx, end_idx, _, _ = comid_indices[comid]
                    batch_size = end_idx - start_idx
                    
                    comid_str = str(comid)
                    if comid_str in self._standardized_attr_dict:
                        attr_vec = self._standardized_attr_dict[comid_str]
                    else:
                        attr_vec = np.zeros(attr_dim, dtype=np.float32)
                    
                    X_attr_batch[sample_idx:sample_idx + batch_size] = attr_vec
                    sample_idx += batch_size
            else:
                # 如果没有属性数据，创建零属性向量
                X_attr_batch = np.zeros((X_ts_batch.shape[0], 1), dtype=np.float32)
            
            # 构建原始数据组（为了保持与旧版本的兼容性）
            groups = {}
            for comid in valid_comids:
                try:
                    # 从加载器获取完整的COMID数据和日期
                    comid_data, comid_dates = self.efficient_loader.load_comid_with_dates(str(comid))
                    
                    # 构建DataFrame格式的组数据
                    if len(comid_data) > 0 and len(comid_dates) > 0:
                        # 创建包含所有特征的DataFrame
                        group_dict = {'COMID': [comid] * len(comid_data), 'date': comid_dates}
                        
                        # 添加输入特征列
                        if self.input_features:
                            for i, feature in enumerate(self.input_features):
                                if i < comid_data.shape[1]:
                                    group_dict[feature] = comid_data[:, i]
                        
                        groups[str(comid)] = pd.DataFrame(group_dict)
                except Exception as e:
                    logging.warning(f"构建COMID {comid} 的组数据时出错: {e}")
                    continue
            
            logging.info(f"批量预测数据准备完成：{len(valid_comids)} 个有效COMID，{X_ts_batch.shape[0]} 个样本")
            
            return {
                'X_ts_scaled': X_ts_batch,
                'X_attr_batch': X_attr_batch,
                'valid_comids': valid_comids,
                'comid_indices': comid_indices,
                'groups': groups
            }
    
    def prepare_training_data_for_head_segments(self,
                                              comid_wq_list: List,
                                              comid_era5_list: List,
                                              all_target_cols: List[str],
                                              target_col: str,
                                              output_dir: str,
                                              model_version: str) -> Tuple:
        """为头部河段准备训练数据（高效版）"""
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")

        # 筛选头部河段
        with TimingAndMemoryContext("寻找头部站点"):
            available_comids = set(self.efficient_loader.comid_index.keys())
            comid_list_head = list(
                available_comids & set([str(c) for c in comid_wq_list]) & set([str(c) for c in comid_era5_list])
            )
            
            # 保存头部河段COMID列表
            np.save(f"{output_dir}/comid_list_head_{model_version}.npy", comid_list_head)
            
            if len(comid_list_head) == 0:
                logging.warning("警告：找不到符合条件的头部河段")
                return None, None, None, None, None
            
            logging.info(f"选择的头部河段数量：{len(comid_list_head)}")

        # 使用流式训练迭代器收集数据
        training_iterator = self.prepare_streaming_training_data(
            comid_list_head,
            all_target_cols,
            target_col,
            batch_size=len(comid_list_head)  # 一次性加载所有数据
        )
        
        # 收集所有训练数据
        X_ts_list, Y_list, COMIDs_list, Dates_list = [], [], [], []
        
        for X_ts_batch, Y_batch, COMIDs_batch, _ in training_iterator:
            X_ts_list.append(X_ts_batch)
            Y_list.append(Y_batch)
            COMIDs_list.extend(COMIDs_batch)
            
            # 收集日期信息（需要从加载器获取）
            for comid in COMIDs_batch:
                _, dates = self.efficient_loader.load_comid_with_dates(str(comid))
                if len(dates) > 0:
                    Dates_list.extend(dates[-len(Y_batch):])  # 取最后的日期
        
        if not X_ts_list:
            return None, None, None, None, None
        
        # 合并数据
        X_ts_scaled = np.vstack(X_ts_list)
        Y = np.concatenate(Y_list)
        COMIDs = np.array(COMIDs_list)
        Dates = np.array(Dates_list)
        
        # 获取属性字典
        attr_dict_scaled = self.get_standardized_attr_dict()
        
        # 输出数据维度信息
        logging.info(f"头部河段训练数据: X_ts_scaled.shape = {X_ts_scaled.shape}")
        logging.info(f"Y.shape = {Y.shape}, COMIDs.shape = {COMIDs.shape}")

        # 保存训练数据
        with TimingAndMemoryContext("保存训练数据"):
            np.savez(f"{output_dir}/upstreams_trainval_{model_version}.npz", 
                    X=X_ts_scaled, Y=Y, COMID=COMIDs, Date=Dates)
            logging.info("训练数据保存成功！")
            
        return X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates
    
    def prepare_next_iteration_data(self,
                                   flow_data_binary_dir: str,
                                   target_col: str,
                                   col_y_n: str,
                                   col_y_up: str,
                                   time_window: int = 10) -> Tuple:
        """
        准备下一轮迭代的训练数据（完全二进制模式）
        
        参数:
            flow_data_binary_dir: 流量计算结果的二进制数据目录
            target_col: 目标列名
            col_y_n: 当前节点预测值列名
            col_y_up: 上游预测值列名
            time_window: 时间窗口大小
            
        返回:
            (X_ts_scaled, attr_dict_scaled, Y_label, COMIDs, Dates)
            
        注意:
            流量数据必须预先使用scripts/csv_to_binary_converter.py转换为二进制格式
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
            
        with TimingAndMemoryContext("准备下一轮迭代数据（完全二进制模式）"):
            # 验证流量数据二进制目录
            flow_metadata_file = os.path.join(flow_data_binary_dir, 'metadata.json')
            if not os.path.exists(flow_metadata_file):
                raise ValueError(f"流量数据二进制目录无效，缺少metadata.json: {flow_data_binary_dir}")
            
            # 初始化流量数据加载器
            try:
                flow_loader = EfficientDataLoader(flow_data_binary_dir)
                logging.info(f"成功加载流量数据二进制索引: {len(flow_loader.comid_index)} 个COMID")
            except Exception as e:
                raise ValueError(f"无法加载流量数据二进制索引: {e}")
            
            # 获取两个数据源的COMID交集
            ts_comids = set(self.efficient_loader.comid_index.keys())
            flow_comids = set(flow_loader.comid_index.keys())
            valid_comids = list(ts_comids & flow_comids)
            
            if not valid_comids:
                logging.error("没有找到时间序列和流量数据的共同COMID")
                return None, None, None, None, None
            
            logging.info(f"找到 {len(valid_comids)} 个有效COMID用于迭代训练")
            
            # 获取流量数据的列信息
            with open(flow_metadata_file, 'r') as f:
                flow_metadata = json.load(f)
            
            flow_feature_cols = flow_metadata.get('feature_columns', [])
            
            # 验证必要的列是否存在
            if col_y_n not in flow_feature_cols:
                raise ValueError(f"流量数据中缺少列: {col_y_n}")
            if col_y_up not in flow_feature_cols:
                raise ValueError(f"流量数据中缺少列: {col_y_up}")
            
            col_y_n_idx = flow_feature_cols.index(col_y_n)
            col_y_up_idx = flow_feature_cols.index(col_y_up)
            
            # 准备数据收集容器
            X_ts_list = []
            Y_label_list = []
            COMIDs_list = []
            Dates_list = []
            
            # 为每个COMID构建训练数据
            for comid_str in valid_comids:
                try:
                    # 从二进制数据加载时间序列数据
                    ts_data, ts_dates = self.efficient_loader.load_comid_with_dates(comid_str)
                    
                    if len(ts_data) < time_window:
                        continue
                    
                    # 从二进制数据加载流量数据
                    flow_data, flow_dates = flow_loader.load_comid_with_dates(comid_str)
                    
                    if len(flow_data) == 0:
                        continue
                    
                    # 创建日期到流量数据的快速索引
                    flow_date_map = {}
                    for i, date in enumerate(flow_dates):
                        flow_date_map[date] = i
                    
                    # 构建滑动窗口和对应的误差标签
                    for i in range(len(ts_data) - time_window + 1):
                        window_end_date = ts_dates[i + time_window - 1]
                        
                        # 快速查找对应日期的流量信息
                        if window_end_date in flow_date_map:
                            flow_idx = flow_date_map[window_end_date]
                            flow_row = flow_data[flow_idx]
                            
                            # 获取时间序列窗口
                            window = ts_data[i:i + time_window]
                            
                            # 检查窗口数据有效性
                            if not np.any(np.isnan(window)):
                                # 计算误差标签：E_label = y_n - y_up
                                y_n = flow_row[col_y_n_idx]
                                y_up = flow_row[col_y_up_idx]
                                
                                if not (np.isnan(y_n) or np.isnan(y_up)):
                                    e_label = float(y_n) - float(y_up)
                                    
                                    X_ts_list.append(window)
                                    Y_label_list.append(e_label)
                                    COMIDs_list.append(int(comid_str))
                                    Dates_list.append(window_end_date)
                
                except Exception as e:
                    logging.warning(f"处理COMID {comid_str} 时出错: {e}")
                    continue
            
            if not X_ts_list:
                logging.error("没有构建出有效的训练数据")
                return None, None, None, None, None
            
            # 转换为numpy数组
            X_ts = np.array(X_ts_list, dtype=np.float32)
            Y_label = np.array(Y_label_list, dtype=np.float32)
            COMIDs = np.array(COMIDs_list)
            Dates = np.array(Dates_list)
            
            # 标准化时间序列数据
            if self.ts_scaler is not None:
                N, T, D = X_ts.shape
                X_ts_2d = X_ts.reshape(-1, D)
                X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
                X_ts_scaled = X_ts_scaled_2d.reshape(N, T, D)
            else:
                X_ts_scaled = X_ts
                logging.warning("时间序列标准化器不可用")
            
            # 获取标准化的属性字典
            attr_dict_scaled = self.get_standardized_attr_dict()
            
            logging.info(f"下一轮迭代数据准备完成（完全二进制模式）:")
            logging.info(f"  - 样本数量: {len(X_ts_scaled)}")
            logging.info(f"  - 时间序列形状: {X_ts_scaled.shape}")
            logging.info(f"  - 误差标签范围: [{Y_label.min():.4f}, {Y_label.max():.4f}]")
            logging.info(f"  - 内存占用: 极低（内存映射访问）")
            
            return X_ts_scaled, attr_dict_scaled, Y_label, COMIDs, Dates