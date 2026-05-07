"""
====================================================
Nucleic_protein_res4.0 文件检测与查看工具
====================================================
用法：将本文件保存为 check_dataset.py，放在与 Nucleic_protein_res4.0 同级目录
然后运行：python check_dataset.py
"""

import os
from my_01_study.utils.root_utils import root_path_02


FILE_PATH = f"{root_path_02}/sais/data_16/old_data/Nucleic_protein_res4.0"  # 这只是一个没有后缀的zip文件...


def check_file_header(filepath, nbytes=64):
    """读取文件头部字节，判断文件类型"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None

    file_size = os.path.getsize(filepath)
    print(f"📁 文件路径: {os.path.abspath(filepath)}")
    print(f"📦 文件大小: {file_size / (1024 ** 3):.2f} GB ({file_size:,} bytes)")
    print(f"🔍 读取前 {nbytes} 字节... ")

    with open(filepath, 'rb') as f:
        header = f.read(nbytes)

    # 打印十六进制和 ASCII
    print("=" * 60)
    print("原始头部字节 (Hex + ASCII):")
    print("=" * 60)
    for i in range(0, min(nbytes, len(header)), 16):
        chunk = header[i:i + 16]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}  {hex_str:<48}  {ascii_str}")

    return header


def detect_file_type(header):
    """根据文件头魔数判断类型"""
    print(" " + "=" * 60)
    print("文件类型推断:")
    print("=" * 60)

    # HDF5
    if header[:8] == b'\x89HDF\r\n\x1a\n':
        print("✅ 检测到: HDF5 文件 (.h5 / .hdf5)")
        print("   说明: 常用于存储大规模结构化数据，如蛋白质结构数据集")
        return "hdf5"

    # NumPy
    if header[:6] == b'\x93NUMPY':
        print("✅ 检测到: NumPy 数组文件 (.npy)")
        return "numpy"

    # ZIP (可能为 .npz)
    if header[:4] == b'PK\x03\x04':
        print("✅ 检测到: ZIP 压缩格式 (可能是 .npz / .zip)")
        return "zip"

    # SQLite
    if header[:16] == b'SQLite format 3\x00':
        print("✅ 检测到: SQLite 数据库 (.db / .sqlite)")
        return "sqlite"

    # PyTorch
    if header[:6] == b'\x80\x02\x8a\x0a' or header[:4] == b'\x80\x02\x8a' or b'torch' in header[:100]:
        print("✅ 检测到: PyTorch 序列化文件 (.pt / .pth)")
        return "pytorch"

    # Tar
    if header[:5] in [b'ustar', b'\x00ustar']:
        print("✅ 检测到: TAR 归档文件 (.tar)")
        return "tar"

    # JSON (文本)
    if header[:1] == b'{' or header[:1] == b'[':
        print("✅ 检测到: JSON 文本文件")
        return "json"

    # Pickle
    if header[:2] == b'\x80\x02' or header[:2] == b'\x80\x03' or header[:2] == b'\x80\x04':
        print("✅ 检测到: Python Pickle 文件 (.pkl)")
        return "pickle"

    # LMDB
    if b'mdb' in header[:20].lower():
        print("✅ 检测到: LMDB 数据库")
        return "lmdb"

    print("⚠️ 未识别出常见魔数，可能是自定义二进制格式")
    print("   建议尝试用 numpy / pickle / torch 直接加载测试")
    return "unknown"


def try_load_as_numpy(filepath):
    """尝试作为 numpy / npz 加载"""
    print(" " + "=" * 60)
    print("尝试作为 NumPy 加载...")
    print("=" * 60)
    try:
        import numpy as np
        # 先尝试 npz
        try:
            data = np.load(filepath, allow_pickle=True, mmap_mode='r')
            print("✅ 成功作为 NPZ 加载!")
            if hasattr(data, 'files'):
                print(f"   包含的数组: {data.files}")
                for key in list(data.files)[:5]:
                    arr = data[key]
                    print(f"   - {key}: shape={arr.shape}, dtype={arr.dtype}")
                if len(data.files) > 5:
                    print(f"   ... 还有 {len(data.files) - 5} 个数组")
            return "npz", data
        except:
            pass

        # 再尝试 npy
        arr = np.load(filepath, allow_pickle=True, mmap_mode='r')
        print(f"✅ 成功作为 NPY 加载!")
        print(f"   shape={arr.shape}, dtype={arr.dtype}")
        return "npy", arr
    except Exception as e:
        print(f"❌ NumPy 加载失败: {e}")
        return None, None


def try_load_as_hdf5(filepath):
    """尝试作为 HDF5 加载"""
    print(" " + "=" * 60)
    print("尝试作为 HDF5 加载...")
    print("=" * 60)
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            print("✅ 成功作为 HDF5 打开!")
            print(f"   顶层 keys: {list(f.keys())[:20]}")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"   📄 {name}: shape={obj.shape}, dtype={obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"   📁 {name}/")

            print("; 文件结构 (前 30 项):")
            count = [0]

            def limited_visit(name, obj):
                if count[0] >= 30:
                    return
                print_structure(name, obj)
                count[0] += 1

            f.visititems(limited_visit)
        return "hdf5", None
    except Exception as e:
        print(f"❌ HDF5 加载失败: {e}")
        return None, None


def try_load_as_pytorch(filepath):
    """尝试作为 PyTorch 加载"""
    print(" " + "=" * 60)
    print("尝试作为 PyTorch 加载...")
    print("=" * 60)
    try:
        import torch
        # 使用 weights_only=False 避免 PyTorch 2.6 报错
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print("✅ 成功作为 PyTorch 加载!")
        print(f"   类型: {type(data)}")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())[:20]}")
            for k in list(data.keys())[:5]:
                v = data[k]
                if hasattr(v, 'shape'):
                    print(f"   - {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"   - {k}: type={type(v)}")
        elif hasattr(data, 'shape'):
            print(f"   shape={data.shape}, dtype={data.dtype}")
        return "pytorch", data
    except Exception as e:
        print(f"❌ PyTorch 加载失败: {e}")
        return None, None


def try_load_as_pickle(filepath):
    """尝试作为 Pickle 加载"""
    print(" " + "=" * 60)
    print("尝试作为 Pickle 加载...")
    print("=" * 60)
    try:
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print("✅ 成功作为 Pickle 加载!")
        print(f"   类型: {type(data)}")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())[:20]}")
        elif isinstance(data, list):
            print(f"   列表长度: {len(data)}")
            if len(data) > 0:
                print(f"   第一项类型: {type(data[0])}")
        return "pickle", data
    except Exception as e:
        print(f"❌ Pickle 加载失败: {e}")
        return None, None


def main():
    print("=" * 60)
    print("  Nucleic_protein_res4.0 数据集检测工具")
    print("=" * 60)
    # 打印【当前工作目录】（Python程序运行的目录）
    print("当前工作目录:", os.getcwd())
    # 打印【FILE_PATH的完整绝对路径】（文件实际路径）
    print("FILE_PATH 对应路径:", os.path.abspath(FILE_PATH))

    header = check_file_header(FILE_PATH, nbytes=64)
    if header is None:
        return

    file_type = detect_file_type(header)

    # 根据检测结果优先尝试对应格式
    results = {}

    if file_type == "hdf5":
        t, d = try_load_as_hdf5(FILE_PATH)
        results[t] = d
    elif file_type in ["numpy", "zip"]:
        t, d = try_load_as_numpy(FILE_PATH)
        results[t] = d
    elif file_type == "pytorch":
        t, d = try_load_as_pytorch(FILE_PATH)
        results[t] = d
    elif file_type == "pickle":
        t, d = try_load_as_pickle(FILE_PATH)
        results[t] = d
    else:
        # 未知格式，逐个尝试
        for loader in [try_load_as_numpy, try_load_as_hdf5, try_load_as_pytorch, try_load_as_pickle]:
            t, d = loader(FILE_PATH)
            if t:
                results[t] = d
                break

    print(" " + "=" * 60)
    print("检测完成!")
    print("=" * 60)
    if results:
        print(f"✅ 成功识别格式: {list(results.keys())}")
    else:
        print("❌ 未能识别文件格式，可能需要赛事方提供格式说明")
        print("   建议: 查看文件头部是否有文本标记，或联系组委会")


if __name__ == "__main__":
    main()
pass
