import os

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split


class CustomDataset(DataLoader):
    def __init__(self, root=None, imageAddresses=None, transformer=None ):
        assert  imageAddresses != None, 'empty address is not allowed'
        assert  root != None, 'empty root address is not allowed'
        self.addresses = imageAddresses
        if type(imageAddresses) != str:
            self.create_image_class_tuple()
        else:
            self.create_image_class_tuple_test()
        self.root = root
        self.transformer = transformer

    def create_image_class_tuple(self):
        self.all_categories = sorted(list(set([line.strip().split('/')[0] for line in self.addresses if line.strip().split('/')[0]!= 'BACKGROUND_Google'])))
        self.image_class = [(line.strip().split('/')[1],self.all_categories.index(line.strip().split('/')[0]))  for line in self.addresses if line.strip().split('/')[0]!= 'BACKGROUND_Google']

    def create_image_class_tuple_test(self):
        with open(self.addresses, 'r') as file:
            self.all_categories = sorted(list(set([line.strip().split('/')[0] for line in file if line.strip().split('/')[0]!= 'BACKGROUND_Google'])))

        with open(self.addresses, 'r') as file:
            self.image_class = [(line.strip().split('/')[1],self.all_categories.index(line.strip().split('/')[0]))  for line in file if line.strip().split('/')[0]!= 'BACKGROUND_Google']



    def __len__(self):
        return len(self.image_class)

    def image_to_PIL(self, dir=None, label=None):
        assert dir != None, 'epmty addresss for image is not allowed'
        assert label != None, 'category of image is not specified'
        category = self.all_categories[label]
        with open(os.path.join(self.root, category, dir), 'rb') as image:
            img = Image.open(image)
            return img.convert('RGB')

    def __getitem__(self, index):
        image_dir, label = self.image_class[index]
        image = self.image_to_PIL(image_dir, label)
        if self.transformer is not None:
            image = self.transformer(image)
        return image, label

def split_date(address ='train.txt'):
    with open(address, 'r') as file:
        lines = [line for line in file]
    X_train_text, X_val_text = train_test_split(lines, test_size=0.2, random_state=42)
    return X_train_text, X_val_text