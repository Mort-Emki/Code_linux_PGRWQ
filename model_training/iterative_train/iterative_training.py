"""
iterative_training.py - PG-RWQ 迭代训练模块

该模块实现 PG-RWQ (Physics-Guided Recursive Water Quality) 模型的迭代训练过程。
整合数据处理、模型训练、汇流计算和评估功能，实现完整的训练循环。
"""

import os
import numpy as np
import pandas as pd
import logging
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# 导入项目中的函数
from ...flow_routing_modules import flow_routing_calculation
from ...logging_utils import ensure_dir_exists
from ..gpu_memory_utils import (
    log_memory_usage, 
    TimingAndMemoryContext, 
    MemoryTracker,
    force_cuda_memory_cleanup
)

# 导入自定义组件
from .data_handler import DataHandler
from .model_manager import ModelManager
from .evaluation import ConvergenceChecker, DataValidator, ModelVisualizer
from .utils import (
    check_existing_flow_routing_results,
    create_predictor,
    save_flow_results,
    split_train_val_data
)

def iterative_training_procedure(
    df: pd.DataFrame,
    attr_df: pd.DataFrame,
    input_features: Optional[List[str]] = None,
    attr_features: Optional[List[str]] = None,
    river_info: Optional[pd.DataFrame] = None,
    all_target_cols: List[str] = ["TN", "TP"],
    target_col: str = "TN",
    max_iterations: int = 10,
    epsilon: float = 0.01,
    model_type: str = 'rf',
    model_params: Optional[Dict[str, Any]] = None,
    device: str = 'cuda',
    comid_wq_list: Optional[List] = None,
    comid_era5_list: Optional[List] = None,
    input_cols: Optional[List] = None,
    start_iteration: int = 0,
    model_version: str = "v1",
    flow_results_dir: str = "flow_results",
    model_dir: str = "models",
    reuse_existing_flow_results: bool = True
) -> Optional[Any]:
    """
    PG-RWQ 迭代训练主程序
    
    参数:
        df: 日尺度数据 DataFrame，包含 'COMID'、'Date'、target_col、'Qout' 等字段
        attr_df: 河段属性DataFrame
        input_features: 输入特征列表
        attr_features: 属性特征列表
        river_info: 河段信息 DataFrame，包含 'COMID' 和 'NextDownID'
        all_target_cols: 所有目标参数列表
        target_col: 主目标参数名称，如 "TN"
        max_iterations: 最大迭代次数
        epsilon: 收敛阈值（残差平均值）
        model_type: 模型类型，如 'lstm', 'rf', 'informer' 等
        model_params: 模型超参数字典，包含特定模型类型所需的所有参数
        device: 训练设备
        comid_wq_list: 水质站点COMID列表
        comid_era5_list: ERA5覆盖的COMID列表
        input_cols: 时间序列输入特征列表（如果与input_features不同）
        start_iteration: 起始迭代轮数，0表示从头开始，>0表示从指定轮次开始
        model_version: 模型版本号
        flow_results_dir: 汇流结果保存目录
        model_dir: 模型保存目录
        reuse_existing_flow_results: 是否重用已存在的汇流计算结果
        
    返回:
        训练好的模型对象
    """
    # ======================================================================
    # 1. 初始化设置
    # ======================================================================
    # 确保model_params不为None
    if model_params is None:
        model_params = {}
    
    
    # 初始化内存监控
    memory_tracker = MemoryTracker(interval_seconds=120)
    memory_tracker.start()
    
    # 记录初始内存状态
    if device == 'cuda' and torch.cuda.is_available():
        log_memory_usage("[训练开始] ", level=0)
    
    # 创建结果目录
    output_dir = ensure_dir_exists(flow_results_dir)
    model_save_dir = ensure_dir_exists(model_dir)
    logging.info(f"汇流计算结果将保存至 {output_dir}")
    logging.info(f"模型将保存至 {model_save_dir}")
    
    # 记录训练开始信息
    if start_iteration > 0:
        logging.info(f"从迭代 {start_iteration} 开始，模型版本 {model_version}")
    else:
        logging.info(f"从初始训练（迭代 0）开始，模型版本 {model_version}")
        logging.info('选择头部河段进行初始模型训练')
    
    # 初始化组件
    data_handler = DataHandler()
    
    # 确保必要的参数不为None
    if input_features is None:
        logging.error("输入特征不能为空")
        memory_tracker.stop()
        memory_tracker.report()
        return None
        
    if attr_features is None:
        logging.error("属性特征不能为空")
        memory_tracker.stop()
        memory_tracker.report()
        return None
    
    data_handler.initialize(df, attr_df, input_features, attr_features)
    
    model_manager = ModelManager(model_type, device, model_save_dir)
    
    convergence_checker = ConvergenceChecker(epsilon=epsilon)
    
    data_validator = DataValidator()
    
    # 初始化模型变量
    model: Optional[Any] = None
    
    try:
        # ======================================================================
        # 2. 初始模型训练与首次汇流计算
        # ======================================================================
        if start_iteration == 0:
            # 为头部河段准备标准化训练数据
            if comid_wq_list is None or comid_era5_list is None:
                logging.error("水质站点列表和ERA5站点列表不能为空")
                memory_tracker.stop()
                memory_tracker.report()
                return None
                
            X_ts_scaled, attr_dict_scaled, Y_head, COMIDs_head, _ = data_handler.prepare_training_data_for_head_segments(
                comid_wq_list=comid_wq_list,
                comid_era5_list=comid_era5_list,
                all_target_cols=all_target_cols,
                target_col=target_col,
                output_dir=output_dir,
                model_version=model_version
            )
            
            if X_ts_scaled is None:
                logging.error("无法准备头部河段训练数据，终止训练")
                memory_tracker.stop()
                memory_tracker.report()
                return None
            
            # 确定数据维度，更新模型参数
            _, _, input_dim = X_ts_scaled.shape
            if attr_dict_scaled:
                attr_dim = next(iter(attr_dict_scaled.values())).shape[0]
            else:
                attr_dim = 0
                logging.warning("属性字典为空，设置attr_dim为0")
            
            # 获取构建参数和训练参数
            if model_params is not None:
                build_params = model_params.get('build', {}).copy()
                train_params = model_params.get('train', {}).copy()
            else:
                build_params = {}
                train_params = {}
            
            # 添加缺失的维度参数
            if 'input_dim' not in build_params:
                build_params['input_dim'] = input_dim
            if 'attr_dim' not in build_params:
                build_params['attr_dim'] = attr_dim
            
            # 划分训练集和验证集
            train_val_data = split_train_val_data(X_ts_scaled, Y_head, COMIDs_head)
            
            # 创建或加载初始模型
            initial_model_path = f"{model_save_dir}/model_initial_A0_{model_version}.pth"
            model = model_manager.create_or_load_model(
                build_params=build_params,
                train_params=train_params,
                model_path=initial_model_path,
                attr_dict=attr_dict_scaled,
                train_data=train_val_data
            )
            
            # 在初始模型训练后添加验证
            ModelVisualizer.verify_initial_model(
                model=model,
                data_handler=data_handler,
                model_manager=model_manager,
                comid_wq_list=comid_wq_list,
                all_target_cols=all_target_cols,
                target_col=target_col,
                model_save_dir=model_save_dir,
                model_version=model_version
            )
            

            # 创建批处理函数
            predictor = create_predictor(data_handler, model_manager, all_target_cols, target_col)
            
            # 执行初始汇流计算（或加载已有结果）
            exists, flow_result_path = check_existing_flow_routing_results(0, model_version, output_dir)
            if exists and reuse_existing_flow_results:
                # 如果存在且配置为重用，直接加载已有结果
                # 检查二进制格式是否存在
                binary_flow_dir = os.path.join(output_dir, f"flow_routing_iteration_0_{model_version}_binary")
                if os.path.exists(binary_flow_dir):
                    logging.info(f"发现已存在的汇流计算结果（二进制格式）：{binary_flow_dir}")
                    df_flow = None  # 不加载到内存，后续直接使用二进制格式
                else:
                    with TimingAndMemoryContext("加载已有汇流计算结果"):
                        logging.info(f"发现已存在的汇流计算结果，加载：{flow_result_path}")
                        df_flow = pd.read_csv(flow_result_path)
                        logging.info(f"成功加载汇流计算结果，共 {len(df_flow)} 条记录")
            else:
                # 如果不存在或配置为不重用，执行汇流计算
                with TimingAndMemoryContext("执行初始汇流计算"):
                    # 获取标准化后的完整属性字典
                    attr_dict_all = data_handler.get_standardized_attr_dict()
                    
                    # 初始迭代使用E_save=1来保存E值
                    if river_info is not None:
                        df_flow = flow_routing_calculation(
                            df=df.copy(), 
                            iteration=0, 
                            model_func=predictor.predict_batch_comids,
                            river_info=river_info, 
                            v_f_TN=35.0,
                            v_f_TP=44.5,
                            attr_dict=attr_dict_all,
                            model=model,
                            all_target_cols=all_target_cols,
                            target_col=target_col,
                            attr_df=attr_df,
                            E_save=1,  # 保存E值
                            E_save_path=f"{output_dir}/E_values_{model_version}",
                            E_exist= 1,
                            E_exist_path= f"{output_dir}/E_values_{model_version}",
                            enable_debug= True
                        )
                    else:
                        logging.error("河段信息为空，无法进行初始汇流计算")
                        memory_tracker.stop()
                        memory_tracker.report()
                        return None
                    
                    # 汇流计算完成后验证结果
                    if comid_wq_list is not None and 'df_flow' in locals():
                        ModelVisualizer.verify_flow_results(
                            iteration=0,
                            model_version=model_version,
                            df_flow=df_flow,
                            original_df=df,
                            target_col=target_col,
                            comid_wq_list=comid_wq_list,
                            output_dir=flow_results_dir
                        )

                    # 保存汇流计算结果（CSV + 二进制）
                    _ = save_flow_results(df_flow, 0, model_version, output_dir)
        else:
            # ======================================================================
            # 从指定迭代次数开始（加载已有模型和汇流结果）
            # ======================================================================
            # 加载上一轮迭代的模型
            last_iteration = start_iteration - 1
            
            # 获取模型参数
            if model_params is not None:
                build_params = model_params.get('build', {}).copy()
            else:
                build_params = {}
            
            # 添加缺失的维度参数
            if 'input_dim' not in build_params and input_features:
                build_params['input_dim'] = len(input_features)
            if 'attr_dim' not in build_params and attr_features:
                build_params['attr_dim'] = len(attr_features)
            
            # 加载上一轮模型
            model_path = f"{model_save_dir}/model_A{last_iteration}_{model_version}.pth"
            
            if not os.path.exists(model_path):
                logging.error(f"无法找到上一轮模型: {model_path}")
                memory_tracker.stop()
                memory_tracker.report()
                return None
                
            model = model_manager.create_or_load_model(
                build_params=build_params,
                train_params={},  # 不需要训练参数，只是加载
                model_path=model_path
            )
            
            # 加载上一轮的汇流计算结果
            previous_flow_path = os.path.join(output_dir, f"flow_routing_iteration_{last_iteration}_{model_version}.csv")
            
            if not os.path.exists(previous_flow_path):
                logging.error(f"无法找到上一轮汇流计算结果: {previous_flow_path}")
                memory_tracker.stop()
                memory_tracker.report()
                return None
            
            # 检查二进制格式是否存在
            binary_flow_dir = os.path.join(output_dir, f"flow_routing_iteration_{start_from_iteration-1}_{model_version}_binary")
            if os.path.exists(binary_flow_dir):
                logging.info(f"发现上一轮汇流计算结果（二进制格式）：{binary_flow_dir}")
                df_flow = None  # 不加载到内存，后续直接使用二进制格式
            else:
                with TimingAndMemoryContext("加载上一轮汇流计算结果"):
                    df_flow = pd.read_csv(previous_flow_path)
                logging.info(f"已加载上一轮汇流计算结果: {previous_flow_path}")
        
        # ======================================================================
        # 3. 主迭代训练循环
        # ======================================================================
        for it in range(start_iteration, max_iterations):
            with TimingAndMemoryContext(f"迭代 {it+1}/{max_iterations}"):
                logging.info(f"\n迭代 {it+1}/{max_iterations} 开始")
                
                # 获取当前迭代的列名
                col_y_n = f'y_n_{it}_{target_col}'
                col_y_up = f'y_up_{it}_{target_col}'
                
                # 高效收敛性检查（避免大DataFrame合并）
                y_true, y_pred = _extract_convergence_data_efficiently(
                    data_handler, df_flow, it, target_col, col_y_n, output_dir, model_version
                )
                
                if y_true is None or y_pred is None:
                    logging.error(f"无法提取收敛性检查数据")
                    break
                
                # 检查收敛性
                converged, _ = convergence_checker.check_convergence(y_true, y_pred, it)
                
                # 如果已收敛，跳出迭代循环
                if converged:
                    logging.info(f"迭代 {it+1} 已达到收敛条件，训练结束")
                    break
                
                # ======================================================================
                # 4. 准备下一轮迭代的训练数据
                # ======================================================================
                logging.info("准备下一轮迭代的训练数据")
                # 准备下一轮迭代的训练数据（使用二进制格式）
                # 构建当前迭代的二进制数据目录路径
                current_flow_binary_dir = os.path.join(output_dir, f"flow_routing_iteration_{it}_{model_version}_binary")
                
                if os.path.exists(current_flow_binary_dir):
                    X_ts_iter, attr_dict_iter, Y_label_iter, COMIDs_iter, _ = data_handler.prepare_next_iteration_data(
                        flow_data_binary_dir=current_flow_binary_dir,
                        target_col=target_col,
                        col_y_n=col_y_n,
                        col_y_up=col_y_up
                    )
                else:
                    logging.error(f"二进制流量数据目录不存在: {current_flow_binary_dir}")
                    logging.error("请确保前一轮迭代正确保存了二进制格式的流量数据")
                    X_ts_iter = None
                
                if X_ts_iter is None:
                    logging.error("准备训练数据失败，无法继续迭代")
                    break
                
                # 划分训练集和验证集
                train_val_data = split_train_val_data(X_ts_iter, Y_label_iter, COMIDs_iter)
                
                # ======================================================================
                # 5. 训练/加载迭代模型
                # ======================================================================
                model_path = f"{model_save_dir}/model_A{it+1}_{model_version}.pth"
                
                # 确保有build_params和train_params
                if model_params is not None:
                    build_params = model_params.get('build', {}).copy()
                    train_params = model_params.get('train', {}).copy()
                else:
                    build_params = {}
                    train_params = {}
                
                # 看是否已有模型
                if not os.path.exists(model_path):
                    # 如果没有已有模型，训练新模型
                    logging.info(f"训练迭代 {it+1} 的模型")
                    model = model_manager.create_or_load_model(
                        build_params=build_params,
                        train_params=train_params,
                        model_path=model_path,
                        attr_dict=attr_dict_iter,
                        train_data=train_val_data
                    )
                else:
                    # 如果有已有模型，直接加载
                    logging.info(f"加载已有的迭代 {it+1} 模型: {model_path}")
                    model = model_manager.create_or_load_model(
                        build_params=build_params,
                        train_params={},
                        model_path=model_path
                    )
                
                # ======================================================================
                # 6. 执行新一轮汇流计算
                # ======================================================================
                # 创建更新后的模型预测函数
                ##predictor 已经创建，不需要重新创建


                # 执行新一轮汇流计算（或加载已有结果）
                exists, flow_result_path = check_existing_flow_routing_results(it+1, model_version, output_dir)
                
                if exists and reuse_existing_flow_results:
                    # 如果存在且配置为重用，直接加载已有结果
                    # 检查二进制格式是否存在
                    binary_flow_dir = os.path.join(output_dir, f"flow_routing_iteration_{it+1}_{model_version}_binary")
                    if os.path.exists(binary_flow_dir):
                        logging.info(f"发现迭代 {it+1} 汇流计算结果（二进制格式）：{binary_flow_dir}")
                        df_flow = None  # 不加载到内存，后续直接使用二进制格式
                    else:
                        with TimingAndMemoryContext(f"加载迭代 {it+1} 已有汇流计算结果"):
                            logging.info(f"发现已存在的汇流计算结果，加载：{flow_result_path}")
                            df_flow = pd.read_csv(flow_result_path)
                            logging.info(f"成功加载汇流计算结果，共 {len(df_flow)} 条记录")
                else:
                    # 如果不存在或配置为不重用，执行汇流计算
                    with TimingAndMemoryContext(f"执行迭代 {it+1} 汇流计算"):
                        # 获取标准化后的完整属性字典
                        attr_dict_all = data_handler.get_standardized_attr_dict()
                        
                        if river_info is not None:
                            df_flow = flow_routing_calculation(
                                df=df.copy(), 
                                iteration=it+1, 
                                model_func=predictor.predict_batch_comids,
                                river_info=river_info, 
                                v_f_TN=35.0,
                                v_f_TP=44.5,
                                attr_dict=attr_dict_all,
                                model=model,
                                all_target_cols=all_target_cols,
                                target_col=target_col,
                                attr_df=attr_df,
                                E_save=1,  # 保存E值
                                E_save_path=f"{output_dir}/E_values_{model_version}"
                            )
                        else:
                            logging.error("河段信息为空，无法进行汇流计算")
                            break
                        
                        # 执行新一轮汇流计算后验证结果
                        if df_flow is not None and comid_wq_list is not None:  # 确保df_flow已定义
                            ModelVisualizer.verify_flow_results(
                                iteration=it+1,
                                model_version=model_version,
                                df_flow=df_flow,
                                original_df=df,
                                target_col=target_col,
                                comid_wq_list=comid_wq_list,
                                output_dir=flow_results_dir
                            )
                        
                        # 保存汇流计算结果（CSV + 二进制）
                        _ = save_flow_results(df_flow, it+1, model_version, output_dir)
                
                # ======================================================================
                # 7. 检查数据质量
                # ======================================================================
                # 检查此轮迭代的汇流计算结果的异常值
                if df_flow is not None:
                    logging.info(f"检查迭代 {it+1} 的汇流计算结果质量")
                    is_valid_data, _ = data_validator.check_dataframe_abnormalities(
                        df_flow, it+1, all_target_cols
                    )
                    
                    # 如果数据无效，尝试修复
                    if not is_valid_data:
                        logging.warning(f"迭代 {it+1} 的汇流计算结果包含过多异常值，尝试修复...")
                        
                        # 修复异常值
                        df_flow = data_validator.fix_dataframe_abnormalities(
                            df_flow, it+1, all_target_cols
                        )
                        
                        # 保存修复后的结果
                        fixed_path = os.path.join(output_dir, f"flow_routing_iteration_{it+1}_{model_version}_fixed.csv")
                        df_flow.to_csv(fixed_path, index=False)
                        logging.info(f"修复后的结果已保存至 {fixed_path}")
                    
                    # 验证数据一致性
                    if input_features is not None:
                        is_coherent = data_validator.validate_data_coherence(
                            df, df_flow, input_features, all_target_cols, it+1
                        )
                    else:
                        is_coherent = True
                        logging.warning("输入特征为空，跳过数据一致性检查")
                    
                    if not is_coherent:
                        logging.warning(f"数据一致性检查失败，可能会影响迭代 {it+2} 的训练")
                else:
                    logging.info(f"使用二进制模式，跳过迭代 {it+1} 的DataFrame数据质量检查")
        
        # ======================================================================
        # 8. 完成训练
        # ======================================================================
        # 生成内存报告
        memory_tracker.stop()
        _ = memory_tracker.report()
        
        if device == 'cuda':
            log_memory_usage("[训练完成] ")
        
        # 保存最终模型
        final_iter = min(it+1 if 'it' in locals() else 0, max_iterations)
        _save_final_model(model, final_iter, model_version, model_save_dir)
        
        return model
        
    except Exception as e:
        # 捕获训练过程中的异常
        logging.exception(f"训练过程中发生错误: {str(e)}")
        
        # 清理资源
        if device == 'cuda' and torch.cuda.is_available():
            force_cuda_memory_cleanup()
            
        memory_tracker.stop()
        memory_tracker.report()
        
        return None


def _save_final_model(
    model: Optional[Any], 
    final_iter: int, 
    model_version: str, 
    model_save_dir: str
) -> None:
    """
    保存最终模型的内部函数
    
    参数:
        model: 要保存的模型对象
        final_iter: 最终迭代次数
        model_version: 模型版本
        model_save_dir: 模型保存目录
    """
    if model is None:
        logging.warning("模型为空，无法保存最终模型")
        return
    
    if not hasattr(model, 'save_model'):
        logging.warning(
            f"模型类型 {type(model).__name__} 没有 save_model 方法，"
            f"无法保存最终模型"
        )
        return
    
    try:
        final_model_path = os.path.join(
            model_save_dir, 
            f"final_model_iteration_{final_iter}_{model_version}.pth"
        )
        
        model.save_model(final_model_path)
        logging.info(f"最终模型已成功保存至: {final_model_path}")
        
    except Exception as e:
        logging.error(
            f"保存最终模型时发生错误: {str(e)}"
        )


def _extract_convergence_data_efficiently(
    data_handler, 
    df_flow, 
    iteration: int, 
    target_col: str, 
    col_y_n: str, 
    output_dir: str, 
    model_version: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    高效提取收敛性检查数据，避免大DataFrame合并
    
    参数:
        data_handler: 数据处理器
        df_flow: 流量计算结果DataFrame（可能为None）
        iteration: 当前迭代次数
        target_col: 目标列名
        col_y_n: 预测值列名
        output_dir: 输出目录
        model_version: 模型版本
        
    返回:
        (y_true, y_pred): 真实值和预测值数组
    """
    try:
        # 优先使用二进制格式
        binary_flow_dir = os.path.join(output_dir, f"flow_routing_iteration_{iteration}_{model_version}_binary")
        
        if os.path.exists(binary_flow_dir):
            # 从二进制格式提取数据
            from ...efficient_data_loader import EfficientDataLoader
            
            # 加载流量计算结果的二进制数据
            flow_loader = EfficientDataLoader(binary_flow_dir)
            
            # 获取流量数据的列信息
            import json
            flow_metadata_file = os.path.join(binary_flow_dir, 'metadata.json')
            with open(flow_metadata_file, 'r') as f:
                flow_metadata = json.load(f)
            
            flow_feature_cols = flow_metadata.get('feature_columns', [])
            
            if col_y_n not in flow_feature_cols:
                logging.warning(f"流量数据中缺少列: {col_y_n}")
                return None, None
            
            col_y_n_idx = flow_feature_cols.index(col_y_n)
            
            # 获取两个数据源的COMID交集
            ts_comids = set(data_handler.efficient_loader.comid_index.keys())
            flow_comids = set(flow_loader.comid_index.keys())
            valid_comids = list(ts_comids & flow_comids)
            
            y_true_list = []
            y_pred_list = []
            
            # 为每个COMID提取对应的数据
            for comid_str in valid_comids:
                try:
                    # 从时间序列数据获取真实值
                    ts_data, ts_dates = data_handler.efficient_loader.load_comid_with_dates(comid_str)
                    
                    # 从流量数据获取预测值
                    flow_data, flow_dates = flow_loader.load_comid_with_dates(comid_str)
                    
                    if len(ts_data) == 0 or len(flow_data) == 0:
                        continue
                    
                    # 找到时间序列数据中目标列的索引
                    ts_feature_cols = data_handler.input_features
                    if target_col not in ts_feature_cols:
                        continue
                    
                    target_col_idx = ts_feature_cols.index(target_col)
                    
                    # 创建日期到数据的映射
                    ts_date_map = {date: i for i, date in enumerate(ts_dates)}
                    flow_date_map = {date: i for i, date in enumerate(flow_dates)}
                    
                    # 找到共同的日期
                    common_dates = set(ts_dates) & set(flow_dates)
                    
                    for date in common_dates:
                        ts_idx = ts_date_map[date]
                        flow_idx = flow_date_map[date]
                        
                        y_true_val = ts_data[ts_idx, target_col_idx]
                        y_pred_val = flow_data[flow_idx, col_y_n_idx]
                        
                        if not (np.isnan(y_true_val) or np.isnan(y_pred_val)):
                            y_true_list.append(y_true_val)
                            y_pred_list.append(y_pred_val)
                
                except Exception as e:
                    logging.warning(f"处理COMID {comid_str} 时出错: {e}")
                    continue
            
            if y_true_list and y_pred_list:
                return np.array(y_true_list), np.array(y_pred_list)
            else:
                logging.warning("未能从二进制数据提取收敛性检查数据")
                return None, None
        
        else:
            # 回退到DataFrame方式（兼容性）
            if df_flow is None:
                logging.warning("df_flow为None且二进制数据不存在，无法进行收敛性检查")
                return None, None
            
            logging.info("使用DataFrame进行收敛性检查（内存占用较高）")
            
            # 这里仍然需要原始数据df，但我们尽量减少内存占用
            # 只合并必要的列
            required_cols = ['COMID', 'date', col_y_n]
            if col_y_n not in df_flow.columns:
                logging.error(f"df_flow中缺少列: {col_y_n}")
                return None, None
            
            # 从原始二进制数据中采样一些数据进行收敛性检查
            sample_size = min(10000, len(df_flow))  # 限制样本大小
            df_flow_sample = df_flow.sample(n=sample_size)[required_cols]
            
            # 构建对应的真实值数据
            y_true_list = []
            y_pred_list = []
            
            for _, row in df_flow_sample.iterrows():
                comid = str(row['COMID'])
                date = row['date']
                y_pred = row[col_y_n]
                
                if comid in data_handler.efficient_loader.comid_index:
                    try:
                        ts_data, ts_dates = data_handler.efficient_loader.load_comid_with_dates(comid)
                        if date in ts_dates:
                            date_idx = ts_dates.index(date)
                            target_col_idx = data_handler.input_features.index(target_col)
                            y_true = ts_data[date_idx, target_col_idx]
                            
                            if not (np.isnan(y_true) or np.isnan(y_pred)):
                                y_true_list.append(y_true)
                                y_pred_list.append(y_pred)
                    except:
                        continue
            
            if y_true_list and y_pred_list:
                return np.array(y_true_list), np.array(y_pred_list)
            else:
                return None, None
    
    except Exception as e:
        logging.error(f"提取收敛性检查数据时出错: {e}")
        return None, None