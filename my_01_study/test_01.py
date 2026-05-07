import os

from my_01_study.utils.root_utils import root_path_02

FILE_PATH = f"{root_path_02}/sais/data_16/old_data/Nucleic_protein_res4.0.zip"  # ...

def main():
    print("=" * 60)
    print("  Nucleic_protein_res4.0.zip 数据集检测工具")
    print("=" * 60)
    # /home/develop_workspaces/study_03/my_301_SAISfold  # 工作目录...
    # 打印【当前工作目录】（Python程序运行的目录）
    print("当前工作目录:", os.getcwd())
    # 打印【FILE_PATH的完整绝对路径】（文件实际路径）
    print("FILE_PATH 对应路径:", os.path.abspath(FILE_PATH))

if __name__ == "__main__":
    main()
pass
