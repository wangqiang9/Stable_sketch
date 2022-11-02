from torch.utils import data
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import os
import numpy as np
import torch

# helpers functions
def cycle(dl):
    while True:
        for data in dl:
            yield data

# fscoco datasets
class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root, text_root):
        super(TripleDataset, self).__init__()

        self.tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # tranform rgb to sketch
        self.sketch_tranform = transforms.Compose([transforms.functional.rgb_to_grayscale])

        classes, class_to_idx = self.find_classes(photo_root)

        self.photo_root = photo_root
        self.sketch_root = sketch_root
        self.text_root = text_root

        self.photo_paths = sorted(self.make_dataset(self.photo_root))
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.len = len(self.photo_paths)

    def __getitem__(self, index):

        photo_path = self.photo_paths[index]
        sketch_path, label, text = self._getrelate_sketch(photo_path)

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')

        P = self.tranform(photo)
        S = self.tranform(sketch)
        # S = self.sketch_tranform(S) # tranform rgb to gray
        L = label
        T = text
        return {'P': P, 'S': S, 'L': L, 'T': T}

    def __len__(self):
        return self.len

    def make_dataset(self, root):
        images = []
        cnames = os.listdir(root)
        for cname in cnames:
            c_path = os.path.join(root, cname)
            if os.path.isdir(c_path):
                fnames = os.listdir(c_path)
                for fname in fnames:
                    path = os.path.join(c_path, fname)
                    images.append(path)
        return images

    def find_classes(self, root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idex = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idex

    def _getrelate_sketch(self, photo_path):

        paths = photo_path.split('/')
        fname = paths[-1].split('.')[0]
        cname = paths[-2]
        label = self.class_to_idx[cname]
        sketchs = sorted(os.listdir(os.path.join(self.sketch_root, cname)))
        sketch_rel = []
        for sketch_name in sketchs:
            if sketch_name.split('.')[0] == fname:
                sketch_rel.append(sketch_name)
        rnd = np.random.randint(0, len(sketch_rel))
        sketch = sketch_rel[rnd]
        # load text
        text_sigle = sketch.split('.')[0] + '.txt'
        sketch_path = os.path.join(self.sketch_root, cname, sketch)
        text_path = os.path.join(self.text_root, cname, text_sigle)
        f = open(text_path)
        text = f.read()
        # text remove "\n" and "."
        text = text.replace(".", "").replace("\n", "")
        f.close()
        return sketch_path, label, text


photo_root = "/root/sketchimage/fscoco-main/fscoco/fscoco/images"
sketch_root = "/root/sketchimage/fscoco-main/fscoco/fscoco/raster_sketches"
text_root = "/root/sketchimage/fscoco-main/fscoco/fscoco/text"
ds = TripleDataset(photo_root=photo_root, sketch_root=sketch_root, text_root=text_root)
dl = cycle(data.DataLoader(ds, batch_size=4, drop_last=True, shuffle=True, pin_memory=True))

if __name__ == "__main__":
    index = 0
    while True:
        index += 1
        data = next(dl)
        data_image = data["P"]
        data_sketch = data["S"]
        data_label = data["L"]
        data_text = data["T"]
        print(data_image.size(), data_sketch.size(), data_label, data_text)
        # save_data = torch.cat((data_image, data_sketch), dim=0)
        # utils.save_image(save_data, f"./test/{data_text}.png")
