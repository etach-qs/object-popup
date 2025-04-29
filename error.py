import trimesh
from pathlib import Path

def check_ply_files(root_dir, output_txt="bad_files.txt"):
    root_path = Path(root_dir)
    bad_files = []
    
    for ply_file in root_path.glob("**/object.ply"):  # 递归查找所有 object.ply
        try:
            print(f"尝试加载: {ply_file}")
            mesh = trimesh.load(str(ply_file), process=True)  # 尝试加载
            assert mesh is not None  # 确保返回有效网格
            print(f"✓ 成功加载: {ply_file}")
        except Exception as e:
            print(f"✗ 加载失败: {ply_file} -> {str(e)}")
            bad_files.append(str(ply_file))  # 保存为字符串路径
    
    # 将损坏文件列表写入txt文件
    with open(output_txt, "w") as f:
        for file_path in bad_files:
            f.write(f"{file_path}\n")
    
    print(f"\n损坏文件已保存至: {output_txt}")
    return bad_files

# 用法：替换为你的数据目录路径
# bad_files = check_ply_files(
#     "/ssd1/lishujia/object-popup/data/preprocessed/behave_smplh",
#     output_txt="bad_ply_files.txt"  # 可自定义输出文件名
# )
data = trimesh.load("/ailab/user/lishujia-hdd/object_popup_behave/behave_smplh/Sub04_train/toolbox_lift_Date07/t00296/object.ply", process=True) 
import pdb; pdb.set_trace()