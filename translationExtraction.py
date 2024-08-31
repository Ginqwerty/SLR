import csv
import os

# 读取文件夹名字的顺序文件
def read_folder_names(text_file_path):
    with open(text_file_path, 'r') as file:
        folder_names = [line.strip() for line in file.readlines()]
    return folder_names

# 读取CSV文件并将name和对应的德语翻译提取为字典
def read_csv_file(csv_file_path):
    translations_dict = {}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for row in reader:
            # 将name和对应的translation放入字典中
            translations_dict[row['name']] = row['translation']
    return translations_dict

# 根据文件夹名字顺序提取对应的德语翻译
def extract_translations_in_order(folder_names, translations_dict):
    translations_in_order = []
    for folder_name in folder_names:
        if folder_name in translations_dict:
            translations_in_order.append(translations_dict[folder_name])
        else:
            translations_in_order.append(f"Translation not found for: {folder_name}")
    return translations_in_order

# 保存提取的德语翻译到文本文件
def save_translations_to_file(translations, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for translation in translations:
            file.write(translation + '\n')

# 主函数
if __name__ == '__main__':
    # 文件路径
    folder_names_file = '/Users/yimingni/Desktop/ASL Project/SLR/processing_training.txt'  # 文件夹名字顺序文件路径
    csv_file_path = '/Users/yimingni/Desktop/ASL Project/SLR/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'      # CSV文件路径
    output_file_path = '/Users/yimingni/Desktop/ASL Project/SLR/trainingTranslation.txt'  # 保存提取翻译的路径

    # 读取文件夹名字和CSV数据
    folder_names = read_folder_names(folder_names_file)
    translations_dict = read_csv_file(csv_file_path)

    # 根据顺序提取翻译
    translations_in_order = extract_translations_in_order(folder_names, translations_dict)

    # 保存提取的翻译
    save_translations_to_file(translations_in_order, output_file_path)

    print(f"Translations have been saved to {output_file_path}")
