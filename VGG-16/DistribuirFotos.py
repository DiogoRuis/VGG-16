import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2):

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_size = int(len(images) * train_ratio)
        val_size = int(len(images) * val_ratio)
        
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))

        for image in val_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))

        for image in test_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(test_dir, class_name, image))


source_directory = 'dados'
train_directory = 'data'
validation_directory = 'validacao'
test_directory = 'test'

split_data(source_directory, train_directory, validation_directory, test_directory)
