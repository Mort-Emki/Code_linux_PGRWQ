"""
data_handler.py - 高效数据处理与标准化模块

仅支持基于内存映射的二进制数据访问，实现真正的O(1)随机访问
"""

import numpy as np
import pandas as pd
import logging
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
        stats = self.efficient_loader.get_statistics()
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
        
        for X_ts_batch, Y_batch, COMIDs_batch, batch_comids in training_iterator:
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