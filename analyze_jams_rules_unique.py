import os
import json

def find_jams_files(root_dir):
    jams_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jams'):
                jams_files.append(os.path.join(root, file))
    return jams_files

def get_rule_signature(jams_file):
    with open(jams_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        scaper_info = data['annotations'][0]['sandbox']['scaper']
        fg_spec = scaper_info.get('fg_spec', None)
        bg_spec = scaper_info.get('bg_spec', None)
        # 用字符串表示方便比较
        return str(fg_spec), str(bg_spec)
    except Exception as e:
        return None, None

if __name__ == "__main__":
    jams_root = "FSDSoundScapes/jams/train"
    jams_files = find_jams_files(jams_root)
    print(f"Found {len(jams_files)} jams files.")

    unique_rules = set()
    for jams_file in jams_files:
        fg_spec, bg_spec = get_rule_signature(jams_file)
        if fg_spec is not None and bg_spec is not None:
            unique_rules.add((fg_spec, bg_spec))

    print(f"\nUnique (fg_spec, bg_spec) rule combinations: {len(unique_rules)}\n")
    for i, (fg, bg) in enumerate(unique_rules):
        print(f"Rule #{i+1}:")
        print(f"  Foreground (fg_spec): {fg}")
        print(f"  Background (bg_spec): {bg}\n") 