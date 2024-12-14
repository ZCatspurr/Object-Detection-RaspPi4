import os

dirs = ['./1_dataprep/data_images/test', 'C:/Users/4alte/Desktop/yolo_detect/Notes/1_dataprep/data_images/train']

for txt_files in dirs:
    for file in os.listdir(txt_files):
        if file.endswith('.txt'):
            file_pth = os.path.join(txt_files, file)
            with open(file_pth, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = line.split()[0]
                    if class_id != '0': 
                        print(f"Invalid class ID {class_id} in file: {file_pth}")