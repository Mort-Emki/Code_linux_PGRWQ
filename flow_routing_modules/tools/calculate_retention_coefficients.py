"""
calculate_retention_coefficients.py - 计算并保存保留系数R (简化版，只记录必要变量)

改进内容：
1. 只记录计算必需的核心变量
2. 去掉不必要的中间变量计算
3. 保持验证功能但减少存储开销
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import os
import warnings

# 导入已有的计算函数
from ..core.geometry import get_river_length, calculate_river_width
from ..physics.environment_param import compute_retainment_factor
from ...data_processing import load_daily_data, load_river_attributes, detect_and_handle_anomalies


def calculate_retention_coefficients_simplified(
    daily_data_path: str,
    attr_data_path: str,
    output_dir: str = "output",
    parameters: List[str] = ["TN", "TP"],
    v_f_TN: float = 35.0,
    v_f_TP: float = 44.5,
    enable_anomaly_check: bool = True,
    fix_anomalies: bool = True,
    max_records_per_param: int = 10000  # 每个参数的最大记录数量
):
    """
    计算并保存保留系数R，只记录必要变量（TN和TP分开存储）
    
    参数:
        daily_data_path: 日尺度数据文件路径
        attr_data_path: 河段属性数据文件路径  
        output_dir: 输出目录路径
        parameters: 要计算的水质参数列表
        v_f_TN: TN的吸收速率参数 (m/yr)
        v_f_TP: TP的吸收速率参数 (m/yr)
        enable_anomaly_check: 是否启用异常值检测
        fix_anomalies: 是否修复检测到的异常值
        max_records_per_param: 每个参数的最大输出记录数，防止文件过大
    
    返回:
        dict: 包含每个参数的DataFrame的字典
    """
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 加载数据
    print("加载数据...")
    df = load_daily_data(daily_data_path)
    attr_df = load_river_attributes(attr_data_path)
    
    # 确保日期列为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"原始日尺度数据形状: {df.shape}")
    print(f"原始属性数据形状: {attr_df.shape}")
    
    # 2. 异常值检测和修复（与主工作流一致）
    if enable_anomaly_check:
        print("=" * 60)
        print("开始数据质量检查（与主工作流一致）")
        print("=" * 60)
        
        # 2.1 检查流量数据
        print("1. 检查流量数据 (Qout)...")
        df, qout_results = detect_and_handle_anomalies(
            df, 
            columns_to_check=['Qout'], 
            fix_negative=fix_anomalies,
            fix_outliers=False, ##流量数据不进行上下界的异常值修复，因为大江大河干流的流量可能会非常大
            fix_nan=fix_anomalies,
            negative_replacement=0.001,
            nan_replacement=0.001,
            outlier_method='iqr',
            outlier_threshold=8,
            verbose=True,
            data_type='timeseries',
            logger=logging
        )
        
        # 2.2 检查温度数据（如果存在）
        temp_cols = [col for col in ['temperature_2m_mean', 'temperature'] if col in df.columns]
        if temp_cols:
            print("2. 检查温度数据...")
            df, temp_results = detect_and_handle_anomalies(
                df,
                columns_to_check=temp_cols,
                check_negative=False,  # 温度可能为负
                fix_outliers=fix_anomalies,
                fix_nan=fix_anomalies,
                negative_replacement=0.0,  # 保持负温度
                nan_replacement=15.0,  # 默认15°C
                outlier_method='iqr',
                outlier_threshold=8,  # 温度使用更严格的阈值
                verbose=True,
                data_type='timeseries',
                logger=logging
            )
        
        # 2.3 检查属性数据
        print("3. 检查河段属性数据...")
        attr_check_cols = ['lengthkm']  # 只检查长度数据
        available_attr_cols = [col for col in attr_check_cols if col in attr_df.columns]
        if available_attr_cols:
            attr_df, attr_results = detect_and_handle_anomalies(
                attr_df,
                columns_to_check=available_attr_cols,
                check_negative=True,
                fix_negative=fix_anomalies,
                fix_outliers=False,
                fix_nan=fix_anomalies,
                negative_replacement=1.0,  # 默认长度1km
                nan_replacement=1.0,  # 默认长度1km
                outlier_method='iqr',
                outlier_threshold=4,
                verbose=True,
                data_type='attributes',
                logger=logging
            )
        
        print("=" * 60)
        print("数据质量检查完成")
        print("=" * 60)
    
    # 3. 构建河网拓扑和属性字典
    print("构建河网拓扑...")
    
    # 创建NextDownID映射
    topo_dict = attr_df.set_index('COMID')['NextDownID'].to_dict()
    
    # 构建属性字典
    attr_dict = {}
    for _, row in attr_df.iterrows():
        comid_int = int(row['COMID'])
        attr_dict[str(comid_int)] = {
            'lengthkm': row.get('lengthkm', 1.0)
        }

    print(f"河段数量: {len(attr_dict)}")

    # 4. 计算保留系数（只记录必要变量）
    print("计算保留系数（只记录必要变量）...")
    
    # 为每个参数存储结果
    parameter_results = {param: [] for param in parameters}
    
    # 统计计算进度
    total_comids = df['COMID'].nunique()
    processed_count = 0
    success_counts = {param: 0 for param in parameters}
    record_counts = {param: 0 for param in parameters}
    
    # 按河段分组处理
    for comid, group in df.groupby('COMID'):
        processed_count += 1
    
        # 检查是否所有参数都已达到记录限制
        if all(record_counts[param] >= max_records_per_param for param in parameters):
            print(f"所有参数都已达到最大记录数限制 ({max_records_per_param})，停止处理")
            break
            
        if processed_count % 50 == 0:
            print(f"处理进度: {processed_count}/{total_comids} ({processed_count/total_comids*100:.1f}%)")
        
        # 获取下游河段ID
        next_down_id = topo_dict.get(comid, 0)
        next_down_id = int(next_down_id) if next_down_id else 0
        
        # 跳过没有下游的河段（终端河段）
        if next_down_id == 0:
            continue
            
        # 检查下游河段是否有数据
        down_data = df[df['COMID'] == next_down_id]
        if down_data.empty:
            continue
            
        # 按日期排序
        group = group.sort_values('date')
        down_data = down_data.sort_values('date')
        
        # 找到共同的日期
        common_dates = set(group['date']) & set(down_data['date'])
        if not common_dates:
            continue
            
        common_dates = sorted(common_dates)
        
        # 获取共同日期的数据
        up_subset = group[group['date'].isin(common_dates)].set_index('date')
        down_subset = down_data[down_data['date'].isin(common_dates)].set_index('date')
        
        # 计算河道宽度
        up_subset['width'] = calculate_river_width(up_subset['Qout'])
        down_subset['width'] = calculate_river_width(down_subset['Qout'])
        
        # 获取河段长度
        length_up = get_river_length(comid, attr_dict)
        length_down = get_river_length(next_down_id, attr_dict)
        
        # 获取温度数据（如果有）
        temperature = None
        temp_cols = [col for col in ['temperature_2m_mean', 'temperature'] if col in up_subset.columns]
        temp_value = None
        if temp_cols:
            temperature = up_subset[temp_cols[0]]
            temp_value = up_subset[temp_cols[0]]
        
        # 为每个参数和每个日期计算保留系数
        for date in common_dates:
            # 获取当天的数据
            try:
                up_data = up_subset.loc[date]
                down_data_day = down_subset.loc[date]
            except KeyError:
                continue
            
            # 基础数据
            Q_up = up_data['Qout']
            Q_down = down_data_day['Qout']
            W_up = up_data['width']
            W_down = down_data_day['width']
            temp_day = temp_value.loc[date] if temp_value is not None else None
            
            # 为每个参数计算
            for param in parameters:
                # 检查该参数是否已达到记录限制
                if record_counts[param] >= max_records_per_param:
                    continue
                    
                try:
                    # 选择相应的吸收速率
                    v_f = v_f_TN if param == "TN" else v_f_TP
                    
                    # 获取氮浓度数据（仅对TN有效）
                    N_concentration_day = None
                    if param == "TN" and param in up_data.index:
                        N_concentration_day = up_data[param]
                    
                    # 计算单个时间点的保留系数
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
                        

                        # 调用保留系数计算函数
                        R_value = compute_retainment_factor(
                            v_f=v_f,
                            Q_up=pd.Series([Q_up]),
                            Q_down=pd.Series([Q_down]),
                            W_up=pd.Series([W_up]),
                            W_down=pd.Series([W_down]),
                            length_up=length_up,
                            length_down=length_down,
                            temperature=pd.Series([temp_day]) if temp_day is not None else None,
                            N_concentration=pd.Series([N_concentration_day]) if N_concentration_day is not None else None,
                            parameter=param
                        )
                    # 处理结果
                    if hasattr(R_value, 'iloc'):
                        R_final = R_value.iloc[0]
                    else:
                        R_final = R_value
                    
                    # # 后处理：处理可能的NaN值和异常值
                    # if pd.isna(R_final):
                    #     R_final = 0.5
                    # R_final = max(0.0, min(1.0, R_final))  # 限制在[0,1]范围内
                    
                    # 数据质量判断
                    data_quality = 'good' if all([Q_up > 0, Q_down > 0, W_up > 0, W_down > 0]) else 'poor'
                    
                    # 保存简化的记录（只保留必要变量）
                    simplified_record = {
                        # 基本信息
                        'COMID': comid,
                        'NextDownID': next_down_id,
                        'date': date,
                        'parameter': param,
                        
                        # 主要结果
                        f'R_{param}': R_final,
                        
                        # 核心输入参数（用于验证和调试）
                        'v_f': v_f,
                        'Q_up_m3s': Q_up,
                        'Q_down_m3s': Q_down,
                        'W_up_m': W_up,
                        'W_down_m': W_down,
                        'length_up_km': length_up,
                        'length_down_km': length_down,
                        
                        # 环境参数
                        'temperature_C': temp_day,
                        'N_concentration_mgl': N_concentration_day if param == "TN" else None,
                        
                        # 数据质量标志
                        'data_quality_flag': data_quality
                    }
                    
                    parameter_results[param].append(simplified_record)
                    record_counts[param] += 1
                    success_counts[param] += 1
                    
                except Exception as e:
                    print(f"计算河段 {comid} 日期 {date} 参数 {param} 时出错: {e}")
                    continue

    print(f"处理完成:")
    for param in parameters:
        print(f"  {param}: 成功计算 {success_counts[param]} 条记录")

    # 5. 为每个参数分别保存结果
    print(f"保存结果到 {output_dir}...")
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_dataframes = {}
    
    for param in parameters:
        if not parameter_results[param]:
            print(f"警告: {param} 没有计算出任何保留系数")
            continue
        
        # 转换为DataFrame
        param_df = pd.DataFrame(parameter_results[param])
        
        # 按COMID和日期排序
        param_df = param_df.sort_values(['COMID', 'date'])
        
        # 生成输出文件路径
        output_file = os.path.join(output_dir, f"retention_coefficients_{param}_simplified.csv")
        
        # 保存到CSV文件
        param_df.to_csv(output_file, index=False)
        
        # 存储到结果字典
        result_dataframes[param] = param_df
        
        # 打印该参数的统计信息
        print(f"\n{param} 保留系数文件: {output_file}")
        print(f"  记录数: {len(param_df)}")
        print(f"  涉及河段: {param_df['COMID'].nunique()} 个")
        print(f"  时间范围: {param_df['date'].min()} 到 {param_df['date'].max()}")
        
        # 保留系数统计
        r_col = f'R_{param}'
        r_values = param_df[r_col].dropna()
        if len(r_values) > 0:
            print(f"  {param} 保留系数统计:")
            print(f"    平均值: {r_values.mean():.4f}")
            print(f"    标准差: {r_values.std():.4f}")
            print(f"    范围: {r_values.min():.4f} - {r_values.max():.4f}")
            
            # 数据质量统计
            good_quality = param_df[param_df['data_quality_flag'] == 'good']
            print(f"    高质量数据比例: {len(good_quality)/len(param_df):.3f}")

    return result_dataframes


def create_simplified_summary(result_dataframes: Dict[str, pd.DataFrame], output_dir: str):
    """
    为每个参数创建简化汇总表
    
    参数:
        result_dataframes: 包含每个参数DataFrame的字典
        output_dir: 输出目录
    """
    print("创建简化汇总表...")
    
    for param, param_df in result_dataframes.items():
        if param_df.empty:
            continue
            
        summary_records = []
        
        # 按河段汇总该参数的数据
        for comid, group in param_df.groupby('COMID'):
            r_col = f'R_{param}'
            
            summary_record = {
                'COMID': comid,
                'parameter': param,
                'record_count': len(group),
                'date_range_days': (group['date'].max() - group['date'].min()).days,
                
                # R值统计
                f'{r_col}_mean': group[r_col].mean(),
                f'{r_col}_std': group[r_col].std(),
                f'{r_col}_min': group[r_col].min(),
                f'{r_col}_max': group[r_col].max(),
                
                # 核心输入数据统计
                'Q_up_mean': group['Q_up_m3s'].mean(),
                'Q_down_mean': group['Q_down_m3s'].mean(),
                'temperature_mean': group['temperature_C'].mean(),
                'length_up_km': group['length_up_km'].iloc[0],  # 长度不变
                'length_down_km': group['length_down_km'].iloc[0],
                
                # 数据质量
                'good_quality_ratio': len(group[group['data_quality_flag'] == 'good']) / len(group),
            }
            
            summary_records.append(summary_record)
        
        # 保存该参数的汇总表
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_output = os.path.join(output_dir, f"retention_coefficients_{param}_summary_simplified.csv")
            summary_df.to_csv(summary_output, index=False)
            
            print(f"  {param} 简化汇总表已保存到: {summary_output}")
    
    return True


import datetime
if __name__ == "__main__":
    # 配置参数
    daily_data_path = 'data/feature_daily_ts.csv'
    attr_data_path = 'data/river_attributes_new.csv'
    output_dir = "output/retention_coefficients_simplified"
    
    ##记录时间
    start_time = datetime.datetime.now()
    print(f"程序开始时间: {start_time}")

    try:
        # 计算简化的保留系数（TN和TP分开存储）
        result_dataframes = calculate_retention_coefficients_simplified(
            daily_data_path=daily_data_path,
            attr_data_path=attr_data_path,
            output_dir=output_dir,
            parameters=["TN", "TP"],
            v_f_TN=35.0,
            v_f_TP=44.5,
            enable_anomaly_check=True,
            fix_anomalies=True,
            max_records_per_param=10000  # 每个参数的最大记录数
        )
        
        # 创建简化汇总表
        if result_dataframes:
            create_simplified_summary(result_dataframes, output_dir)
        
        print("\n" + "="*60)
        print("程序执行完成!")
        end_time = datetime.datetime.now()
        print(f"程序结束时间: {end_time}")
        print(f"程序耗时: {end_time - start_time}")
        
        print("="*60)
        print(f"输出目录: {output_dir}")
        print("\n生成的文件:")
        for param in ["TN", "TP"]:
            print(f"  - retention_coefficients_{param}_simplified.csv  (简化时序数据)")
            print(f"  - retention_coefficients_{param}_summary_simplified.csv  (简化汇总表)")
        
        print("\n简化版特点:")
        print("1. 只记录计算必需的核心变量")
        print("2. 去掉了不必要的中间变量计算")
        print("3. 减少了存储开销，提高了处理效率")
        print("4. 保留了验证所需的基本信息")
        print("5. 保持了数据质量标志用于后续筛选")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()