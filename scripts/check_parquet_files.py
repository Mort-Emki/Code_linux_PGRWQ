#!/usr/bin/env python3
"""
检查parquet文件脚本 - scripts/check_parquet_files.py

从 data/split_daily_data 目录中抽取10个parquet文件并转换为CSV格式进行检查

使用方法:
    python scripts/check_parquet_files.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import os
import random

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def convert_parquet_to_csv(parquet_dir: str, output_dir: str, num_files: int = 10):
    """
    从parquet目录中随机选择文件并转换为CSV
    
    参数:
        parquet_dir: parquet文件目录
        output_dir: 输出CSV文件目录
        num_files: 要转换的文件数量
    """
    parquet_path = Path(parquet_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始检查parquet文件: {parquet_dir}")
    logging.info(f"输出目录: {output_dir}")
    
    # 检查输入目录
    if not parquet_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {parquet_dir}")
    
    # 获取所有parquet文件
    parquet_files = list(parquet_path.glob("daily_COMID_*.parquet"))
    total_files = len(parquet_files)
    
    logging.info(f"找到 {total_files} 个parquet文件")
    
    if total_files == 0:
        raise ValueError("未找到任何parquet文件")
    
    # 随机选择指定数量的文件
    num_files = min(num_files, total_files)
    selected_files = random.sample(parquet_files, num_files)
    
    logging.info(f"随机选择 {num_files} 个文件进行转换")
    
    # 转换文件并收集统计信息
    conversion_stats = []
    successful_conversions = 0
    
    for i, parquet_file in enumerate(tqdm(selected_files, desc="转换文件")):
        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_file)
            
            # 构建CSV文件名
            csv_filename = parquet_file.stem + ".csv"
            csv_filepath = output_path / csv_filename
            
            # 保存为CSV
            df.to_csv(csv_filepath, index=False)
            
            # 收集统计信息
            file_stats = {
                'original_file': parquet_file.name,
                'csv_file': csv_filename,
                'rows': len(df),
                'columns': len(df.columns),
                'parquet_size_mb': parquet_file.stat().st_size / (1024 * 1024),
                'csv_size_mb': csv_filepath.stat().st_size / (1024 * 1024)
            }
            
            conversion_stats.append(file_stats)
            successful_conversions += 1
            
            # 显示前几行数据作为样本
            if i == 0:
                logging.info(f"\n样本数据 ({parquet_file.name}):")
                logging.info(f"形状: {df.shape}")
                logging.info(f"列名: {list(df.columns)}")
                logging.info("前5行数据:")
                logging.info(df.head().to_string())
                
                # 检查数据类型
                logging.info("\n数据类型:")
                for col, dtype in df.dtypes.items():
                    logging.info(f"  {col}: {dtype}")
                
                # 检查缺失值
                missing_counts = df.isnull().sum()
                if missing_counts.sum() > 0:
                    logging.info("\n缺失值统计:")
                    for col, count in missing_counts.items():
                        if count > 0:
                            logging.info(f"  {col}: {count}")
                else:
                    logging.info("\n✓ 无缺失值")
            
        except Exception as e:
            logging.error(f"转换文件 {parquet_file.name} 时出错: {e}")
            continue
    
    # 保存转换统计信息
    if conversion_stats:
        stats_df = pd.DataFrame(conversion_stats)
        stats_path = output_path / "conversion_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        
        # 计算总体统计
        total_parquet_size = stats_df['parquet_size_mb'].sum()
        total_csv_size = stats_df['csv_size_mb'].sum()
        avg_rows = stats_df['rows'].mean()
        avg_cols = stats_df['columns'].mean()
        
        logging.info("=" * 60)
        logging.info("转换完成!")
        logging.info("=" * 60)
        logging.info(f"✓ 成功转换: {successful_conversions}/{num_files} 个文件")
        logging.info(f"✓ 输出目录: {output_path}")
        logging.info(f"✓ 统计文件: {stats_path}")
        logging.info(f"✓ 平均行数: {avg_rows:.0f}")
        logging.info(f"✓ 平均列数: {avg_cols:.0f}")
        logging.info(f"✓ Parquet总大小: {total_parquet_size:.2f} MB")
        logging.info(f"✓ CSV总大小: {total_csv_size:.2f} MB")
        logging.info(f"✓ 大小比率: {total_csv_size/total_parquet_size:.2f}x")
        
        # 显示转换的文件列表
        logging.info("\n转换的文件:")
        for _, row in stats_df.iterrows():
            logging.info(f"  {row['original_file']} -> {row['csv_file']} "
                        f"({row['rows']:.0f}行, {row['parquet_size_mb']:.2f}MB -> {row['csv_size_mb']:.2f}MB)")
    
    return output_path, conversion_stats

def main():
    """主函数"""
    # 设置随机种子以便重现结果
    random.seed(42)
    
    # 设置日志
    setup_logging()
    
    # 设置路径
    parquet_dir = "data/split_daily_data"
    output_dir = "data/csv_samples"
    
    try:
        # 执行转换
        output_path, stats = convert_parquet_to_csv(
            parquet_dir=parquet_dir,
            output_dir=output_dir,
            num_files=10
        )
        
        logging.info(f"\n检查完成! CSV文件已保存到: {output_path}")
        logging.info("你可以打开这些CSV文件来检查数据内容和格式。")
        
    except Exception as e:
        logging.error(f"检查过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()