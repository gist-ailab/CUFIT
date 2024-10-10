import torch.utils.data as data
import os

from PIL import Image

class APTOS2019():
    def __init__(self, root_dir, train = True, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transforms
        
        self.label_txt = os.path.join(root_dir, 'train_1.csv' if train else 'test.csv')
            
        self.samples = []
        with open(self.label_txt, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.split(',')
                if len(line) == 2:
                    img_name, label = line
                    
                    img_name = os.path.join(root_dir, 'train_images/train_images' if train else 'test_images', img_name+'.png')
                    label = label.replace('\n', '')
                    label = int(label)

                    self.samples.append([img_name, label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample = Image.open(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

if __name__ == '__main__':
    aptos = APTOS2019('./data/APTOS-2019', True)
    print(aptos[0])