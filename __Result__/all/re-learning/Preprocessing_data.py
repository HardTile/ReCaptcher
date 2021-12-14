import numpy as np
import os
from sklearn.utils import shuffle
from pathlib import Path

path_base = r"D:\re-learning" # Путь к папке, в которой будет лежать папка с картинками под именем base


# Разделение имен файлов на тестовые и тренировочные для каждого класса картинок, но пока что без переноса самих файлов
os.chdir("base")
list_main_folders = ["train", "validation"]
list_class_folders = os.listdir()

os.chdir("..")

if not os.path.isdir("data"):
    os.mkdir("data")

os.chdir("data")

for folder in list_main_folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)

for folder in list_main_folders:
    os.chdir(folder)
    for class_folder in list_class_folders:
        if not os.path.isdir(class_folder):
            os.mkdir(class_folder)
    os.chdir("..")
    
os.chdir(path_base)
os.chdir("base")

dict_classes = {}
for _class in list_class_folders:
    dict_classes[_class] = None
    
for folder in list_class_folders:
    os.chdir(folder)
    dict_classes[folder] = os.listdir()
    os.chdir("..")
    
for _class in list_class_folders:
    dict_classes[_class] = shuffle(dict_classes[_class], random_state=42)
    
train_file_dict = dict_classes.copy()
val_file_dict = dict_classes.copy()

for _class in list_class_folders:
    train_count = int(len(dict_classes[_class]) * 0.7)
    train_file_dict[_class] = dict_classes[_class][:train_count]
    val_file_dict[_class] = dict_classes[_class][train_count:]
    print(f"{_class}:\ntrain - {len(train_file_dict[_class])}\nvalidation - {len(val_file_dict[_class])}\n")

#Перенос файлов в соответствующие папки с предобработкой
def preprocess_image(_image):
    from PIL import Image
    
    crop_image = _image.crop((20, 30, 110, 90))
    crop_image = crop_image.resize((128, 128), Image.ANTIALIAS)
    
    return crop_image

def crop_and_save(dict_with_class, path_for_save=None):
    from PIL import Image
    
    assert path_for_save is not None, "Specify the path to save"
    
    for _class in list_class_folders:
        i = 0
        for name in dict_with_class[_class]:
            i += 1
            save_name = _class + "_" + str(i)
            img = Image.open(file_source / _class / name)
            crop = preprocess_image(img)
            crop.save(path_for_save / _class / f"{save_name}.png", "PNG")
            
        print(f"Class - {_class} | complete")
        print(f"Total files: {len(os.listdir(path=path_for_save / _class))}")
    
file_source = Path(path_base) / "base"
train_file_destination = Path(path_base) / "data" / "train"
val_file_destination = Path(path_base) / "data" / "validation"

crop_and_save(train_file_dict, path_for_save=train_file_destination)
crop_and_save(val_file_dict, path_for_save=val_file_destination)

print("Data has been successfully prepared!")