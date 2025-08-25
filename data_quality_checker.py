"""
data_quality_checker.py - 统一的数据质量检查模块

提供标准化的数据异常检测和修复功能，消除项目中的重复代码。

⚠️ 重要说明：抽样检测 vs 完整检测
=======================================

1. 完整数据检测 (comprehensive_data_quality_check)
   - 检测完整数据集，可选择修复异常
   - 用于：regression_main.py, main.py 等完整数据处理场景
   - 修复范围：整个数据集
   
2. 抽样数据检测 (sample_based_data_quality_check)  
   - 仅检测抽样数据质量，不修复抽样数据（避免数据不一致）
   - 用于：run_efficient_training.py 等高效训练场景
   - 时间序列：仅检测（样本修复无意义）
   - 属性数据：可修复（完整数据集）
   - 目的：快速评估数据质量，为后续处理提供参考

3. 批处理数据检测 (用于CSV转换)
   - 检测每个数据块，可选择修复
   - 用于：scripts/csv_to_binary_converter.py
   - 修复范围：当前处理的数据块
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from .data_processing import detect_and_handle_anomalies


class DataQualityConfig:
    """数据质量检查配置"""
    
    # 标准参数配置
    QOUT_CHECK = {
        'columns_to_check': ['Qout'],
        'negative_replacement': 0.001,
        'nan_replacement': 0.001,
        'outlier_method': 'iqr',
        'outlier_threshold': 4.0,  # 统一阈值
        'data_type': 'timeseries'
    }
    
    INPUT_FEATURES_CHECK = {
        'check_negative': False,  # 输入特征可能为负
        'negative_replacement': 0.0,
        'nan_replacement': 0.0,
        'outlier_method': 'iqr',
        'outlier_threshold': 6.0,
        'data_type': 'timeseries'
    }
    
    TARGET_DATA_CHECK = {
        'check_nan': False,
        'fix_nan': False,  # 水质数据不填充NaN
        'negative_replacement': 0.001,
        'outlier_method': 'iqr',
        'outlier_threshold': 6.0,
        'data_type': 'timeseries'
    }
    
    ATTR_DATA_CHECK = {
        'check_negative': False,
        'negative_replacement': 0.001,
        'nan_replacement': 0.001,
        'outlier_method': 'iqr',
        'outlier_threshold': 4.0,
        'data_type': 'attributes'
    }


class DataQualityChecker:
    """统一的数据质量检查器"""
    
    def __init__(self, fix_anomalies: bool = False, verbose: bool = True, 
                 logger: Optional[Any] = None):
        """
        初始化数据质量检查器
        
        Args:
            fix_anomalies: 是否修复检测到的异常
            verbose: 是否输出详细日志
            logger: 日志记录器
        """
        self.fix_anomalies = fix_anomalies
        self.verbose = verbose
        self.logger = logger or logging
        
    def check_qout_data(self, df: pd.DataFrame, 
                       exclude_comids: Optional[List] = None,
                       data_type: str = 'timeseries') -> Tuple[pd.DataFrame, Dict]:
        """
        检查流量数据(Qout)
        
        Args:
            df: 数据框
            exclude_comids: 要排除的COMID列表
            data_type: 数据类型标识
            
        Returns:
            (处理后的数据框, 检查结果)
        """
        config = DataQualityConfig.QOUT_CHECK.copy()
        config.update({
            'fix_negative': self.fix_anomalies,
            'fix_outliers': self.fix_anomalies,
            'fix_nan': self.fix_anomalies,
            'verbose': self.verbose,
            'logger': self.logger,
            'exclude_comids': exclude_comids,
            'data_type': data_type
        })
        
        return detect_and_handle_anomalies(df, **config)
    
    def check_input_features(self, df: pd.DataFrame, 
                           input_features: List[str],
                           exclude_comids: Optional[List] = None,
                           data_type: str = 'timeseries') -> Tuple[pd.DataFrame, Dict]:
        """
        检查输入特征数据
        
        Args:
            df: 数据框
            input_features: 输入特征列名列表
            exclude_comids: 要排除的COMID列表
            data_type: 数据类型标识
            
        Returns:
            (处理后的数据框, 检查结果)
        """
        # 检查可用特征
        available_features = [col for col in input_features if col in df.columns]
        if not available_features:
            self.logger.warning("未找到可检查的输入特征列")
            return df, {'has_anomalies': False}
        
        config = DataQualityConfig.INPUT_FEATURES_CHECK.copy()
        config.update({
            'columns_to_check': available_features,
            'fix_nan': self.fix_anomalies,
            'verbose': self.verbose,
            'logger': self.logger,
            'exclude_comids': exclude_comids,
            'data_type': data_type
        })
        
        return detect_and_handle_anomalies(df, **config)
    
    def check_target_data(self, df: pd.DataFrame, 
                         target_cols: List[str],
                         exclude_comids: Optional[List] = None,
                         data_type: str = 'timeseries') -> Tuple[pd.DataFrame, Dict]:
        """
        检查水质目标数据
        
        Args:
            df: 数据框
            target_cols: 目标数据列名列表
            exclude_comids: 要排除的COMID列表
            data_type: 数据类型标识
            
        Returns:
            (处理后的数据框, 检查结果)
        """
        # 检查可用目标列
        available_cols = [col for col in target_cols if col in df.columns]
        if not available_cols:
            self.logger.warning("未找到可检查的水质目标列")
            return df, {'has_anomalies': False}
        
        config = DataQualityConfig.TARGET_DATA_CHECK.copy()
        config.update({
            'columns_to_check': available_cols,
            'verbose': self.verbose,
            'logger': self.logger,
            'exclude_comids': exclude_comids,
            'data_type': data_type
        })
        
        return detect_and_handle_anomalies(df, **config)
    
    def check_attr_data(self, attr_df: pd.DataFrame, 
                       attr_features: List[str],
                       exclude_comids: Optional[List] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        检查河段属性数据
        
        Args:
            attr_df: 属性数据框
            attr_features: 属性特征列名列表
            exclude_comids: 要排除的COMID列表
            
        Returns:
            (处理后的数据框, 检查结果)
        """
        # 检查可用属性特征
        available_features = [col for col in attr_features if col in attr_df.columns]
        if not available_features:
            self.logger.warning("未找到可检查的属性特征列")
            return attr_df, {'has_anomalies': False}
        
        config = DataQualityConfig.ATTR_DATA_CHECK.copy()
        config.update({
            'columns_to_check': available_features,
            'fix_nan': self.fix_anomalies,
            'verbose': self.verbose,
            'logger': self.logger,
            'exclude_comids': exclude_comids
        })
        
        return detect_and_handle_anomalies(attr_df, **config)
    
    def comprehensive_check(self, df: pd.DataFrame, 
                          attr_df: pd.DataFrame,
                          input_features: List[str],
                          target_cols: List[str], 
                          attr_features: List[str],
                          exclude_comids: Optional[List] = None,
                          data_type: str = 'timeseries') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        综合数据质量检查，替代所有重复调用
        
        Args:
            df: 时间序列数据框
            attr_df: 属性数据框
            input_features: 输入特征列表
            target_cols: 目标数据列表
            attr_features: 属性特征列表
            exclude_comids: 要排除的COMID列表
            data_type: 数据类型标识
            
        Returns:
            (处理后的时间序列数据框, 处理后的属性数据框, 综合检查报告)
        """
        quality_report = {}
        
        self.logger.info("开始综合数据质量检查...")
        
        # 1. 检查流量数据
        self.logger.info("1. 检查流量数据(Qout)...")
        df, qout_results = self.check_qout_data(df, exclude_comids, data_type)
        quality_report['qout'] = qout_results
        
        # 2. 检查输入特征
        if input_features:
            self.logger.info("2. 检查输入特征数据...")
            df, input_results = self.check_input_features(df, input_features, exclude_comids, data_type)
            quality_report['input_features'] = input_results
        
        # 3. 检查水质目标数据
        if target_cols:
            self.logger.info("3. 检查水质目标数据...")
            df, target_results = self.check_target_data(df, target_cols, exclude_comids, data_type)
            quality_report['target_data'] = target_results
        
        # 4. 检查属性数据
        if attr_features:
            self.logger.info("4. 检查河段属性数据...")
            attr_df, attr_results = self.check_attr_data(attr_df, attr_features, exclude_comids)
            quality_report['attr_data'] = attr_results
        
        self.logger.info("数据质量检查完成")
        return df, attr_df, quality_report


# 便利函数，保持向后兼容
def check_qout_data(df: pd.DataFrame, fix_anomalies: bool = False, 
                   verbose: bool = True, logger: Optional[Any] = None,
                   exclude_comids: Optional[List] = None,
                   data_type: str = 'timeseries') -> Tuple[pd.DataFrame, Dict]:
    """便利函数：检查流量数据"""
    checker = DataQualityChecker(fix_anomalies, verbose, logger)
    return checker.check_qout_data(df, exclude_comids, data_type)


def check_input_features(df: pd.DataFrame, input_features: List[str],
                        fix_anomalies: bool = False, verbose: bool = True,
                        logger: Optional[Any] = None,
                        exclude_comids: Optional[List] = None,
                        data_type: str = 'timeseries') -> Tuple[pd.DataFrame, Dict]:
    """便利函数：检查输入特征"""
    checker = DataQualityChecker(fix_anomalies, verbose, logger)
    return checker.check_input_features(df, input_features, exclude_comids, data_type)


def check_target_data(df: pd.DataFrame, target_cols: List[str],
                     fix_anomalies: bool = False, verbose: bool = True,
                     logger: Optional[Any] = None,
                     exclude_comids: Optional[List] = None,
                     data_type: str = 'timeseries') -> Tuple[pd.DataFrame, Dict]:
    """便利函数：检查目标数据"""
    checker = DataQualityChecker(fix_anomalies, verbose, logger)
    return checker.check_target_data(df, target_cols, exclude_comids, data_type)


def check_attr_data(attr_df: pd.DataFrame, attr_features: List[str],
                   fix_anomalies: bool = False, verbose: bool = True,
                   logger: Optional[Any] = None,
                   exclude_comids: Optional[List] = None) -> Tuple[pd.DataFrame, Dict]:
    """便利函数：检查属性数据"""
    checker = DataQualityChecker(fix_anomalies, verbose, logger)
    return checker.check_attr_data(attr_df, attr_features, exclude_comids)


def comprehensive_data_quality_check(df: pd.DataFrame, 
                                   attr_df: pd.DataFrame,
                                   input_features: List[str],
                                   target_cols: List[str], 
                                   attr_features: List[str],
                                   fix_anomalies: bool = False,
                                   verbose: bool = True,
                                   logger: Optional[Any] = None,
                                   exclude_comids: Optional[List] = None,
                                   data_type: str = 'timeseries') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """便利函数：综合数据质量检查"""
    checker = DataQualityChecker(fix_anomalies, verbose, logger)
    return checker.comprehensive_check(df, attr_df, input_features, target_cols, 
                                     attr_features, exclude_comids, data_type)


def sample_based_data_quality_check(sample_df: Optional[pd.DataFrame],
                                   attr_df: pd.DataFrame,
                                   input_features: List[str],
                                   target_cols: List[str], 
                                   attr_features: List[str],
                                   fix_anomalies: bool = False,
                                   verbose: bool = True,
                                   logger: Optional[Any] = None,
                                   exclude_comids: Optional[List] = None) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Dict]:
    """
    基于抽样数据的质量检查（用于高效训练场景）
    
    ⚠️  重要说明：
    - 时间序列数据：仅检测不修复（因为只是样本，修复样本没有意义）
    - 属性数据：可以修复（因为是完整数据集）
    
    Args:
        sample_df: 抽样的时间序列数据（可能为None）
        attr_df: 完整的属性数据
        input_features: 输入特征列表
        target_cols: 目标数据列表
        attr_features: 属性特征列表
        fix_anomalies: 是否修复异常（仅对属性数据有效）
        verbose: 是否输出详细日志
        logger: 日志记录器
        exclude_comids: 要排除的COMID列表
        
    Returns:
        (原始抽样数据, 处理后的属性数据, 检查报告)
    """
    # 对于抽样数据，强制设为仅检测模式
    sample_checker = DataQualityChecker(fix_anomalies=False, verbose=verbose, logger=logger)
    # 对于属性数据，使用用户指定的修复模式  
    attr_checker = DataQualityChecker(fix_anomalies=fix_anomalies, verbose=verbose, logger=logger)
    
    quality_report = {}
    
    if logger:
        logger.info("开始基于抽样的数据质量检查...")
        logger.info("注意：时间序列数据仅检测不修复（抽样数据），属性数据可选择修复（完整数据）")
    
    # 初始化检查结果
    quality_report = {
        'qout': {'has_anomalies': False, 'sample_based': True},
        'input_features': {'has_anomalies': False, 'sample_based': True}, 
        'target_data': {'has_anomalies': False, 'sample_based': True},
        'attr_data': {'has_anomalies': False, 'sample_based': False}
    }
    
    if sample_df is not None:
        # 1. 检查流量数据（仅检测）
        if 'Qout' in sample_df.columns:
            if logger:
                logger.info("1. 检查流量数据 (Qout) - 基于抽样（仅检测）...")
            try:
                _, qout_results = sample_checker.check_qout_data(
                    sample_df.copy(), exclude_comids, 'timeseries_sample'
                )
                qout_results['sample_based'] = True
                qout_results['sample_size'] = len(sample_df)
                quality_report['qout'] = qout_results
                
                if qout_results.get('has_anomalies', False):
                    if logger:
                        logger.warning("⚠️  抽样数据中发现异常，但不修复抽样数据")
                        logger.info("💡 如需修复，请对完整数据集进行处理")
            except Exception as e:
                if logger:
                    logger.warning(f"流量数据检查出错: {e}")
        else:
            if logger:
                logger.info("1. 跳过流量数据检查 (数据不可用)")
        
        # 2. 检查输入特征（仅检测）
        if input_features:
            if logger:
                logger.info("2. 检查日尺度输入特征 - 基于抽样（仅检测）...")
            try:
                _, input_results = sample_checker.check_input_features(
                    sample_df.copy(), input_features, exclude_comids, 'timeseries_sample'
                )
                input_results['sample_based'] = True
                input_results['sample_size'] = len(sample_df)
                quality_report['input_features'] = input_results
                
                if input_results.get('has_anomalies', False):
                    if logger:
                        logger.warning("⚠️  输入特征抽样数据中发现异常，但不修复抽样数据")
                        logger.info("💡 如需修复，请对完整数据集进行处理")
            except Exception as e:
                if logger:
                    logger.warning(f"输入特征检查出错: {e}")
        else:
            if logger:
                logger.info("2. 跳过输入特征检查 (数据不可用)")
        
        # 3. 检查水质目标数据（仅检测）
        if target_cols:
            if logger:
                logger.info("3. 检查水质目标数据 - 基于抽样（仅检测）...")
            try:
                _, target_results = sample_checker.check_target_data(
                    sample_df.copy(), target_cols, exclude_comids, 'timeseries_sample'
                )
                target_results['sample_based'] = True
                target_results['sample_size'] = len(sample_df)
                quality_report['target_data'] = target_results
                
                if target_results.get('has_anomalies', False):
                    if logger:
                        logger.warning("⚠️  水质目标抽样数据中发现异常，但不修复抽样数据")
                        logger.info("💡 如需修复，请对完整数据集进行处理")
            except Exception as e:
                if logger:
                    logger.warning(f"水质目标数据检查出错: {e}")
        else:
            if logger:
                logger.info("3. 跳过水质目标数据检查 (数据不可用)")
    else:
        if logger:
            logger.info("跳过时间序列数据检查 (无抽样数据)")
    
    # 4. 检查属性数据（完整数据，可以修复）
    if attr_features:
        if logger:
            fix_status = "可修复" if fix_anomalies else "仅检测"
            logger.info(f"4. 检查河段属性数据（完整数据，{fix_status}）...")
        try:
            attr_df, attr_results = attr_checker.check_attr_data(
                attr_df, attr_features, exclude_comids
            )
            attr_results['sample_based'] = False
            quality_report['attr_data'] = attr_results
        except Exception as e:
            if logger:
                logger.warning(f"属性数据检查出错: {e}")
    
    if logger:
        logger.info("基于抽样的数据质量检查完成")
        
        # 总结报告
        anomaly_found = any(report.get('has_anomalies', False) for report in quality_report.values())
        if anomaly_found:
            logger.warning("📊 检查摘要：发现数据异常")
            logger.info("📝 抽样检测结果仅供参考，实际训练将使用完整数据集")
        else:
            logger.info("✅ 检查摘要：未发现明显异常")
    
    return sample_df, attr_df, quality_report