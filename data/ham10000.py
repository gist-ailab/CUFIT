import os
import shutil

def create_folder(folder_name):
    folder = os.path.join('ham10000', folder_name)
    class_list = ['MEL', 'NV','BCC','AKIEC','BKL','DF','VASC']

    os.mkdir(folder)
    for c in class_list:
        folder = os.path.join('ham10000', folder_name, c)
        os.mkdir(folder)

def read_and_move(csv_file, img_folder, is_train=True):
    class_list = ['MEL', 'NV','BCC','AKIEC','BKL','DF','VASC']

    if is_train:
        dst_folder = 'ham10000/train/'
    else:
        dst_folder = 'ham10000/test/'


    with open(csv_file, 'r') as f:
        lines = f.readlines()[1:]

        for line in lines:
            items = line.split(',')
            items[-1] = items[-1].replace('\n', '')
            class_info = [float(x) for x in items[1:]]
            class_info = class_info.index(1.0)

            img = '{}.jpg'.format(items[0])
            label = class_info

            img_path_src = os.path.join(img_folder, img)
            img_path_dst = os.path.join(dst_folder, class_list[label], img)

            shutil.copy2(img_path_src, img_path_dst)


if __name__ == '__main__':
    train_image_folder = 'ISIC2018_Task3_Training_Input'
    train_image_gt_csv = 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

    test_image_folder = 'ISIC2018_Task3_Test_Input'
    test_image_gt_csv = 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'

    os.mkdir('./ham10000')
    # Create train folder
    create_folder('train')
    read_and_move(train_image_gt_csv, train_image_folder)


    # Create test folder
    create_folder('test')
    read_and_move(test_image_gt_csv, test_image_folder, is_train=False)
        