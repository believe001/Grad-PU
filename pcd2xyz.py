import numpy as np

def read_pcd_file(pcd_path):
    """
    一个基础的函数来读取PCD文件。
    假设PCD文件是ASCII格式的，每一行包含一个点的信息。
    这里简化处理，仅提取x, y, z坐标。
    """
    points = []
    with open(pcd_path, 'r') as file:
        is_header = True
        for line in file:
            if is_header:
                if line.startswith('DATA ascii'):
                    is_header = False
            else:
                parts = line.strip().split()
                if len(parts) >= 3:  # 确保至少有x, y, z三个坐标值
                    point = [float(part) for part in parts[:3]]
                    points.append(point)
    return np.array(points)

def convert_pcd_to_xyz(input_pcd_path, output_xyz_path):
    """
    将PCD文件转换为XYZ文件。
    """
    # 读取PCD文件
    points = read_pcd_file(input_pcd_path)

    # 写入XYZ文件
    with open(output_xyz_path, 'w') as file:
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"转换完成，已保存至 {output_xyz_path}")

# 使用示例
input_pcd_file = 'path/to/your/input.pcd'
output_xyz_file = 'path/to/your/output.xyz'
convert_pcd_to_xyz(input_pcd_file, output_xyz_file)