import os
import matplotlib.pyplot as plt
import zipfile

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
file_paths = [
    "./循环伏安原始数据记录文件/Run Unsaved 50mv.txt",
    "./循环伏安原始数据记录文件/Run Unsaved 100mv.txt",
    "./循环伏安原始数据记录文件/Run Unsaved 150mv.txt",
    "./循环伏安原始数据记录文件/Run Unsaved 200mv.txt",
    "./循环伏安原始数据记录文件/Run Unsaved 250mv.txt"
]

# 存储每个电压级别的数据
data_last_two_cycles = {}

# 读取文件数据并绘制图表
def read_voltage_current_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

            # 寻找包含 "Potential/V, Current/A" 的行的索引
            target_string = "Potential/V, Current/A"
            potential_current_index = next((i for i, line in enumerate(lines) if target_string in line), None)

            if potential_current_index is not None:
                # 从找到的索引后一行开始查找包含 "-0.2" 的索引
                found_indices = [i for i in range(potential_current_index + 1, len(lines)) if "-0.2" in lines[i]]

                if len(found_indices) >= 2:
                    # 提取倒数第一圈的电压和电流数据
                    last_circle_start = found_indices[-2] - 1
                    last_circle_end = found_indices[-1]
                    
                    voltages_last_circle1 = []
                    currents_last_circle1 = []
                    
                    for i in range(last_circle_start, last_circle_end):
                        line = lines[i].strip()
                        items = line.split(",")
                        if len(items) >= 2:
                            voltage = float(items[0])
                            current = float(items[1])
                            voltages_last_circle1.append(voltage)
                            currents_last_circle1.append(current)
                    
                    # 提取倒数第二圈的电压和电流数据
                    if len(found_indices) >= 3:  # 检查是否至少有三个循环
                        last_circle_start = found_indices[-3] - 1  # 调整为倒数第二个循环的起始索引
                        last_circle_end = found_indices[-2]  # 使用最后找到的循环的索引
                    
                        voltages_last_circle2 = []
                        currents_last_circle2 = []
                    
                        for i in range(last_circle_start, last_circle_end):
                            line = lines[i].strip()
                            items = line.split(",")
                            if len(items) >= 2:
                                voltage = float(items[0])
                                current = float(items[1])
                                voltages_last_circle2.append(voltage)
                                currents_last_circle2.append(current)
                    
                    # 存储数据
                    voltage_level = os.path.basename(file_path).replace('.txt', '')
                    data_last_two_cycles[voltage_level] = {
                        'voltages_last_circle1': voltages_last_circle1,
                        'currents_last_circle1': currents_last_circle1,
                        'voltages_last_circle2': voltages_last_circle2,
                        'currents_last_circle2': currents_last_circle2
                    }

                    return True  # 表示绘图成功

    except Exception as e:
        print(f"发生错误: {e}")
        return False  # 表示绘图失败

    return False  # 如果未成功绘图

# 处理每个文件路径
for file_path in file_paths:
    read_voltage_current_data(file_path)

# 绘制每个电压级别的倒数两圈数据并保存单独图像
for voltage_level, data in data_last_two_cycles.items():
    voltages_last_circle1 = data['voltages_last_circle1']
    currents_last_circle1 = data['currents_last_circle1']
    voltages_last_circle2 = data['voltages_last_circle2']
    currents_last_circle2 = data['currents_last_circle2']

    # 绘制单独图像
    plt.figure(figsize=(8, 6))
    plt.plot(voltages_last_circle1, currents_last_circle1, marker='o', markersize=0.05, linestyle='-', color='r', label='倒数第一圈')
    plt.plot(voltages_last_circle2, currents_last_circle2, marker='o', markersize=0.05, linestyle='-', color='b', label='倒数第二圈')
    plt.xlabel('电压 (V)', fontsize=12, weight='bold')
    plt.ylabel('电流 (A)', fontsize=12, weight='bold')
    plt.title(f'{voltage_level} 倒数第一圈和倒数第二圈 电压(V) vs电流(A) 曲线', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, which='both')
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.tight_layout()
    plt.minorticks_on()

    # 拟合图存储路径
    save_dir = "./循环伏安拟合图结果"  # 当前目录文件夹
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 构建保存文件路径并保存图像
    filename = os.path.join(save_dir, f'{voltage_level}.png')
    plt.savefig(filename, dpi=300)
    print(f"保存图片：{filename}")

    plt.show()

# 绘制整合图像：所有电压级别的数据绘制在同一坐标系下
plt.figure(figsize=(8, 6))

for voltage_level, data in data_last_two_cycles.items():
    voltages_last_circle1 = data['voltages_last_circle1']
    currents_last_circle1 = data['currents_last_circle1']
    voltages_last_circle2 = data['voltages_last_circle2']
    currents_last_circle2 = data['currents_last_circle2']
    
    plt.plot(voltages_last_circle1, currents_last_circle1, marker='o', markersize=0.05, linestyle='-', label=f'{voltage_level} - 倒数第一圈')
    plt.plot(voltages_last_circle2, currents_last_circle2, marker='o', markersize=0.05, linestyle='-', label=f'{voltage_level} - 倒数第二圈')

plt.xlabel('电压 (V)', fontsize=12, weight='bold')
plt.ylabel('电流 (A)', fontsize=12, weight='bold')
plt.title('所有电压级别的倒数第一圈和倒数第二圈数据 电压(V) vs电流(A) 曲线', fontsize=14, weight='bold')
plt.legend()
plt.grid(True, which='both')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.tight_layout()
plt.minorticks_on()

# 拟合图存储路径
save_dir = "./循环伏安拟合图结果"  # 当前目录文件夹
# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 构建保存文件路径并保存图像
filename = os.path.join(save_dir, '不同电压汇总图.png')
plt.savefig(filename, dpi=300)
print(f"保存图片：{filename}")

plt.show()

# 压缩生成的图片
zip_file_name = "./循环伏安拟合图结果.zip"
with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, save_dir)
            zipf.write(file_path, arcname)

print(f'压缩完成，文件保存为: {zip_file_name}')
