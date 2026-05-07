import os
import torch
print('Device name:', torch.cuda.get_device_name(0))
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)


is_win = 1
if os.name == 'nt':  # Windows系统
    print('Windows系统')
else:
    is_win = 0
    print('Linux系统')

# /mnt/d/develop/PyCharm_workspaces/study_01/my_00_data  # 存储数据的根目录
# linux 直接使用绝对路径更简单
root_path = '/mnt/d/develop/PyCharm_workspaces/study_01/my_00_data'  # 存储数据的根目录

if is_win == 1:
    root_path = '../../../../study_01/my_00_data'  # 存储数据的根目录

