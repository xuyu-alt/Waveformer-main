import os
import json

def find_jams_files(root_dir):
    jams_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jams'):
                jams_files.append(os.path.join(root, file))
    return jams_files

def print_jams_rules(jams_file):
    with open(jams_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        scaper_info = data['annotations'][0]['sandbox']['scaper']
        fg_spec = scaper_info.get('fg_spec', None)
        bg_spec = scaper_info.get('bg_spec', None)
        print(f"\nFile: {jams_file}")
        print("  Foreground (fg_spec):")
        print(f"    {fg_spec}")
        print("  Background (bg_spec):")
        print(f"    {bg_spec}")
    except Exception as e:
        print(f"\nFile: {jams_file}")
        print("  [Warning] Could not extract rules:", e)

if __name__ == "__main__":
    # 修改为你的jams文件根目录
    jams_root = "FSDSoundScapes/jams/train"
    jams_files = find_jams_files(jams_root)
    print(f"Found {len(jams_files)} jams files.")

    for jams_file in jams_files[:5]:  # 只打印前5个，防止输出太多。可去掉[:5]查看全部
        print_jams_rules(jams_file) 