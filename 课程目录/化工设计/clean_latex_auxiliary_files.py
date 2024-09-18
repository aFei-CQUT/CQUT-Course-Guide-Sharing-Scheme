import os
import glob

def clean_latex_files(directory='.'):
    # 定义要删除的文件扩展名列表
    extensions_to_delete = [
        '*.aux', '*.log', '*.out', '*.toc', '*.lof', '*.lot', 
        '*.blg', '*.synctex.gz', '*.fls', '*.fdb_latexmk','*.bbl'
    ]

    # 遍历目录及子目录
    for root, dirs, files in os.walk(directory):
        for ext in extensions_to_delete:
            files_to_delete = glob.glob(os.path.join(root, ext))
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f"已删除: {file}")
                except Exception as e:
                    print(f"删除 {file} 时出错: {e}")

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.getcwd()
    
    # 执行清理
    clean_latex_files(current_dir)
    print("LaTeX辅助文件清理完成。")
