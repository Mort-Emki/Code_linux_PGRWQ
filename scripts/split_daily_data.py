#!/usr/bin/env python3
"""
数据拆分预处理脚本 - scripts/split_daily_data.py

这是一个独立的预处理脚本，用于将大的CSV文件按COMID拆分成小文件。
只需要运行一次，之后训练时直接使用拆分后的文件。

使用方法:
    python scripts/split_daily_data.py --input data/feature_daily_ts.csv --output data/split_daily_data --format parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('split_data.log')
        ]
    )

def split_daily_data_by_comid(csv_path: str, output_dir: str, file_format: str = 'parquet'):
    """
    将大的日尺度数据CSV按COMID拆分成小文件
    
    参数:
        csv_path: 原始CSV文件路径
        output_dir: 输出目录
        file_format: 文件格式 ('parquet', 'csv', 'feather')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始拆分数据: {csv_path}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"文件格式: {file_format}")
    
    # 检查输入文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"输入文件不存在: {csv_path}")
    
    # 检查文件大小
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    logging.info(f"输入文件大小: {file_size_mb:.1f} MB")
    
    # 读取原始数据
    logging.info("正在加载原始数据...")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise
    
    # 检查必要的列
    if 'COMID' not in df.columns:
        raise ValueError("数据中缺少 'COMID' 列")
    
    # 统计信息
    total_comids = df['COMID'].nunique()
    total_records = len(df)
    logging.info(f"总计: {total_comids} 个COMID, {total_records} 条记录")
    
    # 检查日期列
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].agg(['min', 'max'])
        logging.info(f"时间范围: {date_range['min']} 到 {date_range['max']}")
    
    # 按COMID分组
    logging.info("开始按COMID分组...")
    comid_groups = df.groupby('COMID')
    
    # 创建索引字典和统计信息
    comid_index = {}
    file_stats = []
    files_saved = 0
    total_size_mb = 0
    
    # 拆分保存
    logging.info(f"开始拆分并保存 {total_comids} 个COMID的数据...")
    
    for comid, group_data in tqdm(comid_groups, desc="拆分文件", total=total_comids):
        try:
            # 构建文件名
            if file_format == 'parquet':
                filename = f"daily_COMID_{comid}.parquet"
                filepath = output_path / filename
                group_data.to_parquet(filepath, index=False)
            elif file_format == 'csv':
                filename = f"daily_COMID_{comid}.csv"
                filepath = output_path / filename
                group_data.to_csv(filepath, index=False)
            elif file_format == 'feather':
                filename = f"daily_COMID_{comid}.feather"
                filepath = output_path / filename
                group_data.to_feather(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")
            
            # 记录文件信息
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size_mb += file_size_mb
            
            comid_index[str(comid)] = {
                'filename': filename,
                'records': len(group_data),
                'size_mb': file_size_mb
            }
            
            file_stats.append({
                'COMID': comid,
                'filename': filename,
                'records': len(group_data),
                'size_mb': file_size_mb
            })
            
            files_saved += 1
            
        except Exception as e:
            logging.error(f"保存COMID {comid} 数据时出错: {e}")
            continue
    
    # 保存索引文件
    logging.info("保存索引文件...")
    index_df = pd.DataFrame([
        {
            'COMID': comid,
            'filename': info['filename'],
            'records': info['records'],
            'size_mb': info['size_mb']
        }
        for comid, info in comid_index.items()
    ])
    
    index_path = output_path / "comid_index.csv"
    index_df.to_csv(index_path, index=False)
    
    # 保存统计信息
    stats_path = output_path / "split_statistics.json"
    import json
    statistics = {
        'original_file': csv_path,
        'original_size_mb': file_size_mb,
        'split_format': file_format,
        'total_comids': total_comids,
        'files_created': files_saved,
        'total_split_size_mb': total_size_mb,
        'compression_ratio': total_size_mb / file_size_mb if file_format == 'parquet' else 1.0,
        'avg_file_size_mb': total_size_mb / files_saved,
        'avg_records_per_file': total_records / files_saved
    }
    
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # 输出总结
    logging.info("=" * 60)
    logging.info("数据拆分完成!")
    logging.info("=" * 60)
    logging.info(f"✓ 成功拆分: {files_saved} 个文件")
    logging.info(f"✓ 索引文件: {index_path}")
    logging.info(f"✓ 统计文件: {stats_path}")
    logging.info(f"✓ 原始大小: {file_size_mb:.1f} MB")
    logging.info(f"✓ 拆分后大小: {total_size_mb:.1f} MB")
    
    if file_format == 'parquet':
        logging.info(f"✓ 压缩比: {statistics['compression_ratio']:.2f}x")
    
    logging.info(f"✓ 平均文件大小: {statistics['avg_file_size_mb']:.2f} MB")
    logging.info(f"✓ 平均记录数/文件: {statistics['avg_records_per_file']:.0f}")
    
    return output_path, comid_index, statistics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='拆分日尺度数据CSV文件')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    parser.add_argument('--format', '-f', default='parquet', 
                      choices=['parquet', 'csv', 'feather'],
                      help='输出文件格式 (默认: parquet)')
    parser.add_argument('--force', action='store_true', 
                      help='强制覆盖已存在的输出目录')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 检查输出目录
    if os.path.exists(args.output) and not args.force:
        index_file = os.path.join(args.output, 'comid_index.csv')
        if os.path.exists(index_file):
            logging.warning(f"输出目录已存在且包含索引文件: {args.output}")
            response = input("是否继续? (y/N): ")
            if response.lower() != 'y':
                logging.info("操作取消")
                return
    
    try:
        # 执行拆分
        output_path, comid_index, statistics = split_daily_data_by_comid(
            csv_path=args.input,
            output_dir=args.output, 
            file_format=args.format
        )
        
        logging.info("数据拆分预处理完成! 现在可以使用流式训练了。")
        
        # 建议下一步操作
        logging.info("\n下一步操作:")
        logging.info(f"1. 检查输出目录: {output_path}")
        logging.info("2. 运行训练脚本:")
        logging.info(f"   python run_streaming_training.py --split-dir {args.output}")
        
    except Exception as e:
        logging.error(f"拆分过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()