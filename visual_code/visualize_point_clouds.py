import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def visualize_point_cloud(data, output_image_path):
    """可视化点云并保存为图片"""
    plt.figure(figsize=(10, 10), dpi=300)  # 高清输出
    plt.scatter(data[:, 0], data[:, 1], c='lightblue', s=1)  # 绘制点云
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)  # 保存图片
    plt.close()

def visualize_point_clouds(input_folder, output_folder):
    """遍历文件夹中的所有.xyz文件进行可视化"""
    # 获取所有 .xyz 文件
    xyz_files = glob.glob(os.path.join(input_folder, '*.xyz'))

    # 遍历所有文件并可视化
    for file in xyz_files:
        try:
            data = np.loadtxt(file)
            file_name = os.path.basename(file)
            output_image_path = os.path.join(output_folder, f"{file_name}.png")
            visualize_point_cloud(data, output_image_path)
            print(f"Saved: {output_image_path}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize all .xyz files in a folder and save as images.')

    # 设置输入文件夹和输出路径的默认值
    parser.add_argument('--input_folder', type=str, default='/home/lh/code/Grad-PU/pretrained_model/pugan/test/myintpuX', help='Input folder containing .xyz files.')
    parser.add_argument('--output_folder', type=str, default='./scan', help='Folder to save the output images.')

    args = parser.parse_args()

    # 创建输出文件夹，如果不存在
    os.makedirs(args.output_folder, exist_ok=True)

    visualize_point_clouds(args.input_folder, args.output_folder)
