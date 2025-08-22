#!/usr/bin/env python3
"""
check_binary_compatibility.py - 检查数据是否为二进制兼容格式

用于验证现有数据是否已转换为高效流量计算所需的二进制格式
"""

import os
import pandas as pd
import logging
from pathlib import Path

def check_data_compatibility(data_path: str) -> dict:
    """
    检查数据是否兼容新的高效流量计算系统
    
    参数:
        data_path: 数据文件或目录路径
        
    返回:
        检查结果字典
    """
    result = {
        'compatible': False,
        'format': 'unknown',
        'issues': [],
        'recommendations': []
    }
    
    try:
        # 检查是否为文件
        if os.path.isfile(data_path):
            if data_path.endswith('.csv'):
                result['format'] = 'csv'
                result['issues'].append('CSV格式不支持高效计算')
                result['recommendations'].append('请使用scripts/csv_to_binary_converter.py转换为二进制格式')
                
            elif data_path.endswith(('.npz', '.npy')):
                result['format'] = 'numpy'
                result['issues'].append('需要检查是否符合EfficientDataLoader格式要求')
                result['recommendations'].append('确认数据结构包含：data.npy, dates.npy, metadata.json')
                
        # 检查是否为目录
        elif os.path.isdir(data_path):
            # 检查是否为有效的二进制数据目录
            required_files = ['metadata.json', 'data.npy', 'dates.npy']
            missing_files = []
            
            for file_name in required_files:
                file_path = os.path.join(data_path, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
            
            if not missing_files:
                # 验证元数据格式
                import json
                metadata_path = os.path.join(data_path, 'metadata.json')
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    required_fields = ['n_comids', 'n_days', 'feature_columns', 'comid_list']
                    missing_fields = [f for f in required_fields if f not in metadata]
                    
                    if not missing_fields:
                        result['compatible'] = True
                        result['format'] = 'efficient_binary'
                        
                        # 额外信息
                        result['stats'] = {
                            'n_comids': metadata['n_comids'],
                            'n_days': metadata['n_days'],
                            'features': metadata['feature_columns'],
                            'estimated_memory_saving': '约400x内存减少'
                        }
                    else:
                        result['issues'].extend([f'元数据缺少字段: {field}' for field in missing_fields])
                        
                except Exception as e:
                    result['issues'].append(f'元数据读取失败: {e}')
            else:
                result['format'] = 'incomplete_binary'
                result['issues'].extend([f'缺少必要文件: {file}' for file in missing_files])
                result['recommendations'].append('使用csv_to_binary_converter.py重新转换数据')
                
        else:
            result['issues'].append(f'路径不存在: {data_path}')
            
    except Exception as e:
        result['issues'].append(f'检查过程出错: {e}')
        
    return result


def print_compatibility_report(data_path: str):
    """
    打印数据兼容性报告
    
    参数:
        data_path: 数据路径
    """
    print(f"\n=== 数据兼容性检查报告 ===")
    print(f"检查路径: {data_path}")
    print(f"检查时间: {pd.Timestamp.now()}")
    
    result = check_data_compatibility(data_path)
    
    print(f"\n格式类型: {result['format']}")
    print(f"兼容性: {'✅ 兼容' if result['compatible'] else '❌ 不兼容'}")
    
    if result['compatible']:
        print(f"\n✅ 恭喜！您的数据已经是高效二进制格式")
        if 'stats' in result:
            stats = result['stats']
            print(f"   - COMID数量: {stats['n_comids']:,}")
            print(f"   - 时间天数: {stats['n_days']:,}")
            print(f"   - 特征数量: {len(stats['features'])}")
            print(f"   - 内存优化: {stats['estimated_memory_saving']}")
            print(f"   - 特征列表: {', '.join(stats['features'][:5])}{'...' if len(stats['features']) > 5 else ''}")
    else:
        print(f"\n❌ 发现问题:")
        for issue in result['issues']:
            print(f"   - {issue}")
            
        print(f"\n💡 建议:")
        for rec in result['recommendations']:
            print(f"   - {rec}")
        
        if result['format'] == 'csv':
            print(f"\n🚀 转换命令:")
            print(f"   python scripts/csv_to_binary_converter.py --input {data_path} --output {data_path.replace('.csv', '_binary')}")
    
    print("\n" + "="*50)


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查数据格式兼容性')
    parser.add_argument('data_path', help='数据文件或目录路径')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，只返回结果')
    
    args = parser.parse_args()
    
    if args.quiet:
        result = check_data_compatibility(args.data_path)
        exit(0 if result['compatible'] else 1)
    else:
        print_compatibility_report(args.data_path)


if __name__ == '__main__':
    main()