import os
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device name:', torch.cuda.get_device_name(0))


is_win = 1
if os.name == 'nt':  # Windows系统
    print('Windows系统')
else:
    is_win = 0
    print('Linux系统')

# /mnt/d/develop/PyCharm_workspaces/study_01/my_00_data  # 存储数据的根目录
# /home/develop_workspaces/study_03/my_301_SAISfold  # 工作目录...
# linux 直接使用绝对路径更简单
root_path = '/mnt/d/develop/PyCharm_workspaces/study_01/my_00_data/sais/data_16/old_data'  # 存储数据的根目录

if is_win == 1:
    root_path = '../../../../study_01/my_00_data/sais/data_16/old_data'  # 存储数据的根目录

FILE_PATH = f"{root_path}/Nucleic_protein_res4.0.zip"  # ...

def main():
    print("=" * 60)
    print("  Nucleic_protein_res4.0.zip 数据集检测工具")
    print("=" * 60)
    # 打印【当前工作目录】（Python程序运行的目录）
    print("当前工作目录:", os.getcwd())
    # 打印【FILE_PATH的完整绝对路径】（文件实际路径）
    print("FILE_PATH 对应路径:", os.path.abspath(FILE_PATH))

if __name__ == "__main__":
    main()
pass
