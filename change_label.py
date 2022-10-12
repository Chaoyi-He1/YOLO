import os

folder_path = "./my_yolo_dataset/val/labels"
file_list = os.listdir(folder_path)
for fname in file_list:
    file_name = os.path.join(folder_path, fname)
    f = open(file_name, 'r')
    updated_data = ''
    for line in f:
        line = line.split()
        line[0] = str(int(line[0]) - 1)
        line = ' '.join(line) + '\n'
        updated_data = updated_data + line
    with open(file_name, 'w') as label:
        label.write(updated_data)
        label.close()
