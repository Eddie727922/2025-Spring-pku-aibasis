import os
import zipfile


def zipDir(dirpath, outFullName):
    with zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dirpath):
            # 获取相对于源目录的相对路径，并统一使用正斜杠
            rel_path = os.path.relpath(root, dirpath)
            rel_path = rel_path.replace(os.path.sep, "/")

            path_parts = rel_path.split("/")
            if path_parts[0] in {"data", "__pycache__"}:
                continue  # 跳过目标目录下的 data 和 __pycache__ 文件夹及其内容

            for file in files:
                if file.endswith(".zip"):
                    continue  # 跳过ZIP文件
                print(file)
                file_path = os.path.join(root, file)
                arcname = os.path.join(rel_path, file).replace(os.path.sep, "/")
                zipf.write(file_path, arcname)


if __name__ == "__main__":
    # ---------------------------------------------------------
    group = "2"  # 替换为你的组号
    id = "2300010631"  # 替换为你的学号
    name = "吴羽迪"  # 替换为你的姓名
    # ---------------------------------------------------------

    zip_name = f"{group}组-{id}-{name}-hw5.zip"
    current_file_directory_path = os.path.dirname(os.path.abspath(__file__))
    input_path = current_file_directory_path
    output_path = os.path.join(current_file_directory_path, zip_name)

    zipDir(input_path, output_path)
