# this script replaces ' E ' with ' & ' in XML files within a specified directory and its subdirectories.
# but not the 'E' at the beginning or end of a line. In prose-01, it has been skipped. has to manually replace it.
import os
import re

def replace_space_E_space_in_xml(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".xml"):
                file_path = os.path.join(subdir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 替换 ' E ' 为 ' & '
                new_content = re.sub(r' E ', ' & ', content)

                if content != new_content:
                    # 先保存备份
                    backup_path = file_path + ".bak"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # 写入新内容
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

    print("替换完成，所有被修改的文件都有 .bak 备份。")

# 使用方法：替换下面的路径为你的根目录
replace_space_E_space_in_xml("/Users/ziyanxu/AGLAGLA/output_gml_xml/E_removed_output_gml_xml")
