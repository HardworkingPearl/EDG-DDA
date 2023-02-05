# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import numpy as np
import pickle
import pandas as pd

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # EDG
    "EDGCircle",
    "EDGEvolCircle",
    "EDGSine",
    "EDGRotatedMNIST",
    "EDGPortrait",
    "EDGOcularDisease",
    "EDGForestCover",
    "EDGCaltran"
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            if not hasattr(self, 'env_sample_number'):
                'origin'
                images = original_images[i::len(environments)]
                labels = original_labels[i::len(environments)]
            else:
                'specific sample number'
                images = original_images[i *
                                         self.env_sample_number:(i + 1) * self.env_sample_number]
                labels = original_labels[i *
                                         self.env_sample_number:(i + 1) * self.env_sample_number]
            'all data, replicate'
            # images = original_images
            # labels = original_labels
            'all data / 10, replicate'
            # images = original_images[10::10]
            # labels = original_labels[10::10]
            # images = images[10::10]
            # labels = labels[10::10]
            self.datasets.append(dataset_transform(
                images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
                                                         1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST_OLD(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    # ENVIRONMENTS = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
    # ENVIRONMENTS = ['0', '20', '40', '60', '80', '100', '120', '140', '160']

    def __init__(self, root, test_envs, hparams):
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           # [0, 20, 40, 60, 80, 100, 120, 140, 160], # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],  #
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)
        d = angle / 15 * torch.ones_like(y)  # /15
        if self.dm_idx:
            return TensorDataset(x, y, d)
        else:
            return TensorDataset(x, y)


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, tuple(map(lambda x: int(x), self.ENVIRONMENTS)),
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)
        if self.dm_idx:
            d = angle / self.env_dist * torch.ones_like(y)  
            return TensorDataset(x, y, d)
        else:
            return TensorDataset(x, y)


class EDGRotatedMNIST(RotatedMNIST):
    ''' spawn envs based on the input env_density in hparams
    '''
    ENVIRONMENTS = ['0', '5', '10', '15', '20', '25', '30', '35',
                    '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90']

    def __init__(self, root, test_envs, hparams):
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
            self.env_dist = hparams['env_distance']
        else:
            self.dm_idx = False
        # upper = 90
        # ed = hparams['env_density'] # this is density
        # self.ENVIRONMENTS = [str(int((i/(ed-1))*upper)) for i in range(ed)]
        # print(f"EDGRotatedMNIST - the generated {ed} envs are: {self.ENVIRONMENTS}.")
        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(hparams['env_number'])]
        if hparams['total_sample_number'] == 0:
            self.env_sample_number = hparams['env_sample_number']
        else:
            self.env_sample_number = int(hparams['total_sample_number'] / hparams['env_number'])
        print(
            f"EDGRotatedMNIST - the generated {hparams['env_number']} envs are: {self.ENVIRONMENTS}.")
        super(RotatedMNIST, self).__init__(root, tuple(map(lambda x: int(x), self.ENVIRONMENTS)),
                                           self.rotate_dataset, (1, 28, 28,), 10)


class EDGCombineDataset():
    '''
    random sample one from last d then cat with then return
    '''

    def __init__(self, d, last_d):
        self.d = d
        self.last_d = d

    def __getitem__(self, i):
        '''random sample one from the last then stack and return'''
        rand_i = 0

        # res = torch.stack((, self.d[i]), dim=0)
        return {
            'support': self.last_d[rand_i],
            'query': self.d[i],
        }

    def __len__(self):
        return len(self.d)


class EDGPortrait(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False
        # load data
        # self.env_sample_number = hparams['env_sample_number'] # always 200 # maximum env_distance * env_number * env_sample_number
        # self.env_distance = hparams['env_distance']
        # self.env_number = hparams['env_number']
        '(temp) fixed spliting metric'
        # always 200 # maximum env_distance * env_number * env_sample_number
        original_images, original_labels = self.load_portraits_data()

        total_sample_number = len(original_images)
        self.env_number = hparams['env_number']
        self.env_distance = int(total_sample_number / self.env_number)
        self.env_sample_number = int(self.env_distance * hparams['env_sample_ratio'])

        self.ENVIRONMENTS = [str(self.env_distance * i)
                             for i in range(self.env_number)]
        # self.env_sample_number = 200

        # split and append to self.datasets
        self.datasets = []
        domain = 0
        for i in range(len(self.ENVIRONMENTS)):
            'same'
            # images = original_images[:1000]
            # labels = original_labels[:1000]
            '[1000:2000][2000:3000]'  # distance minimum 1
            images = original_images[i * self.env_distance:i * self.env_distance + self.env_sample_number]
            labels = original_labels[i * self.env_distance:i * self.env_distance + self.env_sample_number]
            x_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()])
            x = torch.zeros(len(images), 1, 32, 32)
            for i in range(len(images)):
                x[i] = x_trans(images[i])
            y = torch.tensor(labels).view(-1).long()
            if self.dm_idx:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y, torch.tensor([float(domain)] * len(x))))
            else:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y))
            domain += 1
        self.input_shape = (1, 32, 32,)
        self.num_classes = 2

    def load_portraits_data(self, path='portrait_dataset_32x32.mat'):
        # total sample number 37921, order by year
        from scipy import io
        data = io.loadmat(os.path.join(self.data_dir, path))
        return data['Xs'], data['Ys'][0]

class EDGEvolCircle(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False
        domain_num = 30
        data_pkl = self.load_circle_data(data_dir)
        self.datasets = []
        self.datasets_ = []
        for d in range(domain_num):
            # get x, y from data_pkl
            idx = data_pkl['domain'] == d
            x = data_pkl['data'][idx].astype(np.float32)
            y = data_pkl['label'][idx].astype(np.int64)
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            if self.dm_idx:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y, torch.tensor([float(d)] * len(x))))
                self.datasets_.append((torch.tensor(x).float(), y, torch.tensor([float(d)] * len(x))))
            else:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y))
        self.input_shape = (2,)
        self.num_classes = 2
        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(domain_num)]

    def load_circle_data(self, path):
        return self.read_pickle(os.path.join(path, 'evol_circle/data/evol_circle.pkl'))

    def read_pickle(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data
    


class EDGCircle(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        domain_num = 30
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False

        data_pkl = self.load_circle_data(self.data_dir)
        self.datasets = []
        for d in range(domain_num):
            # get x, y from data_pkl
            idx = data_pkl['domain'] == d
            x = data_pkl['data'][idx].astype(np.float32)
            y = data_pkl['label'][idx].astype(np.int64)
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            if self.dm_idx:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y, torch.tensor([float(d)] * len(x))))
            else:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y))
        self.input_shape = (2,)
        self.num_classes = 2
        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(domain_num)]

    def load_circle_data(self, dir='../datasets_for_domainbed/toy-circle/data/half-circle.pkl'):
        path = os.path.join(dir, 'toy-circle/data/half-circle.pkl')
        return self.read_pickle(path)

    def read_pickle(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data


class EDGSine(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        domain_num = 12
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False
        data_pkl = self.load_circle_data(data_dir)
        self.datasets = []
        data = np.stack(data_pkl['data'], axis=0)
        data = np.reshape(data, [domain_num, len(data) // domain_num, 2])
        ys = np.array(data_pkl['label'])
        ys = np.reshape(ys, [domain_num, len(ys) // domain_num])
        domains = np.array(data_pkl['domain'])
        domains = np.reshape(domains, [domain_num, len(domains) // domain_num])
        for d in range(hparams['env_number']):  # domain_num
            # get x, y from data_pkl
            assert domains[d][0] == d, "domain index not consistent"
            x = data[d]
            y = ys[d]
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            if self.dm_idx:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y, torch.tensor([float(d)] * len(x))))
            else:
                self.datasets.append(TensorDataset(torch.tensor(x).float(), y))
            # self.datasets.append(TensorDataset(torch.tensor(x).float(), y))
        self.input_shape = (2,)
        self.num_classes = 2
        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(domain_num)]

    def load_circle_data(self, path='../datasets_for_domainbed/toy-sine/sine.pkl'):
        return self.read_pickle(os.path.join(path, 'toy-sine/sine.pkl'))

    def generate_sine_data(self, hparams):
        f = lambda x: np.sin(x*np.pi/2 + np.pi/2)
        interval = 4/ hparams['env_number']
        for i in range(hparams['env_number']):
            x0 = np.random.rand("env_sample_number")*interval+ i*interval
            dy0 = np.random.rand("env_sample_number")
            x1 = np.random.rand("env_sample_number") * interval + i * interval
            dy1 = -np.random.rand("env_sample_number")

        np.random.rand(1200)

    def read_pickle(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data


class EDGForestCover(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False

        COL = 'Elevation'
        MAX = 3451  # df[COL].max()
        MIN = 2061  # df[COL].min()
        COUNT = hparams['env_number'] + 1

        # pre
        self.datasets = []
        # df = self.load_forestcover_data().drop('Id', axis = 1)
        df = self.load_forestcover_data()
        # MAX = df[COL].max() # 3451 # df[col].max()
        # MIN = df[COL].min() # 2061 # df[col].min()
        bins = np.arange(MIN, MAX, (MAX - MIN) / COUNT)
        se1 = pd.cut(df[COL], bins)
        df = df.drop(COL, axis=1)
        gb = df.groupby(se1)
        gbs = [gb.get_group(x) for x in gb.groups]
        # groupby('Cover_Type').size()
        for each in gbs:
            print(each.groupby('label').size())
        gbs = [self.get_xy_from_df(each) for each in gbs]
        for i, (x, y) in enumerate(gbs):
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            # print(y)
            # normalize the input x
            x = torch.nn.functional.normalize(torch.tensor(x).float(), dim=0)
            if self.dm_idx:
                self.datasets.append(TensorDataset(x, y, torch.tensor([float(i)] * len(x))))
            else:
                self.datasets.append(TensorDataset(x, y))
        self.input_shape = (54,)
        self.num_classes = 2
        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(COUNT - 1)]
        return

    def load_forestcover_data(self, path='ForestCover/train.csv'):
        df = pd.read_csv(os.path.join(self.data_dir, path))
        df = df.rename(columns={"Cover_Type": "label"})
        df = self.group_labels(df)
        df = df.drop('Id', axis=1)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=False)  # [index, label]
        return df

    def group_labels(self, df):
        groups = [
            [0, 1, 6, 3],
            [4, 5, 2, 7],
        ]

        # print(df)

        def new_label(row):
            for new_l in range(len(groups)):
                if row['label'] in groups[new_l]:
                    return new_l

        df['label'] = df.apply(new_label, axis=1)
        # print(df)
        return df

    def get_xy_from_df(self, df):
        Y = df['label'].to_numpy()
        X = df.drop('label', axis='columns').to_numpy()
        return (X, Y)


class BalancedClassDataset(torch.utils.data.IterableDataset):
    """ BalancedClassDataset
        package data by classes, 
        then each class becomes a TensorDataset.
        At each __iter__, 
        we sample N_p samples from each class. each one with shape (N_p, 1, 28, 28)
        then concat all classes and return
    """

    def __init__(self, x, y, num_classes, N_p):  # TODO add para num_classes
        super(BalancedClassDataset).__init__()
        data_num = x.shape[0]
        dataset = TensorDataset(x, y)
        self.dataloaders = [None for _ in range(num_classes)]
        for each_c in range(num_classes):
            mask = [1 if y[i] == each_c else 0 for i in range(data_num)]
            c_idxs = torch.nonzero(torch.tensor(mask)).flatten()
            sampler = torch.utils.data.SubsetRandomSampler(
                c_idxs, generator=None)
            # see also: torch.utils.data.Subset or torch.utils.data.SubsetRandomSampler(indices, generator=None)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=N_p, sampler=sampler, shuffle=False, num_workers=0)
            self.dataloaders[each_c] = dataloader

    def __iter__(self):
        while True:
            x, y = zip(*[next(iter(dataloader))
                         for dataloader in self.dataloaders])
            x = torch.stack(x, dim=0)
            y = torch.stack(y, dim=0)
            yield (x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.427, 0.328, 0.283], std=[0.258, 0.252, 0.225])
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.427, 0.328, 0.283], std=[0.258, 0.252, 0.225])
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            if self.dm_idx:
                env_dataset = DGImageFolder(path,  # ImageFolder(path,
                                            transform=env_transform, domain_index=torch.tensor(float(i)))
            else:
                env_dataset = ImageFolder(path,  # ImageFolder(path,
                                          transform=env_transform)
            # env_dataset = DGImageFolder(path, # ImageFolder(path,
            #                           transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class DGImageFolder(ImageFolder):
    def __init__(self, path, transform, domain_index):
        super().__init__(path, transform)
        self.domain_index = domain_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.domain_index


class EDGOcularDisease(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [i for i in range(10)]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "OcularDisease/dg/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)

        
        
class EDGCaltran(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = [i for i in range(46)]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "CalTran/dg/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)



class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


def OcularDisease_pre():
    def rename_label(df):
        def new_label(row):
            return row['label'].strip("'][")

        df['label'] = df.apply(new_label, axis=1)
        return df

    def regroup_label(df):
        groups = [
            ['N'],
            ['D'],
            ['A', 'C', 'G', 'M', 'O', 'H']
        ]

        # print(df)

        def new_label(row):
            for new_l in range(len(groups)):
                if row['label'] in groups[new_l]:
                    return new_l

        df['label'] = df.apply(new_label, axis=1)
        # print(df)
        return df

    def cp_image(src, dst):
        # print(src)
        # print(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return

    # INIT
    ROOT = '../datasets_for_domainbed/OcularDisease'
    TABLE_PATH = 'full_df.csv'
    # read data
    df = pd.read_csv(os.path.join(ROOT, TABLE_PATH))
    # pre process
    df = df.rename(columns={'labels': 'label'})
    # label process
    df = rename_label(df)
    df = regroup_label(df)
    # group process
    COL = 'Id'
    COL = 'Patient Age'
    MAX_ = 85  # df[col].max()
    MIN_ = 30  # df[col].min()
    COUNT_ = 11
    ranges = np.arange(MIN_, MAX_, (MAX_ - MIN_) / COUNT_)
    cut_res = pd.cut(df[COL], ranges)
    gb = df.groupby(cut_res)
    for d in range(len(gb.groups.keys())):
        # d - domain; gb_key - domain_range;
        gb_key = list(gb.groups.keys())[d]
        domain_df = gb.get_group(gb_key)
        for index, row in domain_df.iterrows():
            # cp images one by one
            src = os.path.join(ROOT, 'preprocessed_images/', row.filename)
            dst = os.path.join(ROOT, 'dg', str(
                d), str(int(row.label)), row.filename)
            cp_image(src, dst)
            print(dst)
    # print(gb_key)
    # print(gb.get_group(gb_key).groupby('label').size())


def gen_gaussian_data(mean, cov, alpha, num):
    data_X = np.random.multivariate_normal(mean, cov, num)
    # X*alpha^T
    alpha = np.array(alpha)
    data_Y = np.matmul(data_X, alpha.reshape((-1, 1))).reshape((num))
    data_Y = np.sign(data_Y) * 0.5 + 0.5
    return (data_X, data_Y)


def gen_evol_circle():
    tar_path = "../datasets_for_domainbed/evol_circle/data/evol_circle.pkl"
    DOMAIN_NUM = 30
    SAMPLE_NUM_EACH_DOMAIN = 500
    MEAN = [0, 0]
    COV = [
        [1, 0],
        [0, 1]
    ]
    angle_list = [2 * np.pi * i / DOMAIN_NUM for i in range(DOMAIN_NUM)]
    alpha_list = [(np.cos(angle_list[i]), np.sin(angle_list[i])) for i in range(DOMAIN_NUM)]
    res = {
        'data': [],
        'label': [],
        'domain': []
    }
    for d_i in range(DOMAIN_NUM):
        data_X, data_Y = gen_gaussian_data(MEAN, COV, alpha_list[d_i], SAMPLE_NUM_EACH_DOMAIN)
        res['data'].append(data_X)
        res['label'].append(data_Y)
        res['domain'].append(np.array([d_i for _ in range(SAMPLE_NUM_EACH_DOMAIN)]))
    for k, v in res.items():
        res[k] = np.concatenate(v)
    with open(tar_path, 'wb') as f:
        pickle.dump(res, f)
        # np.save(tar_path, res)
    return res

def gen_caltran():
    import scipy.io
    import glob
    import os
    import shutil
    ROOT = 'CalTran'
    
    mat = scipy.io.loadmat('CalTran/caltran_dataset_labels.mat')
    img_names = glob.glob('CalTran/*.jpg')
    img_names.sort()
    
    num_spls_per_domain = 118
    num_domains = len(img_names) // num_spls_per_domain
    
    f = lambda x: 0 if x==-1 else 1
    def cp_image(src, dst):
        # print(src)
        # print(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    
    #for k in range(len(mat['names'])):
    #    if mat['names'][k][0][0] == '2013-03-07-09-21':
    #        print(f"{k}!!")
    
    for i in range(num_domains):
        for j in range(num_spls_per_domain):
            index = i*num_spls_per_domain+j        
            src = os.path.join(ROOT, mat['names'][index][0][0]+".jpg")  # img_names[index].split("/")[1])
            if img_names[index].split("/")[1][:-4] == mat['names'][index][0][0]:                
                dst = os.path.join(ROOT, 'dg', str(i), str(f(mat['labels'][0,index])), mat['names'][index][0][0]+".jpg")
            else:
                for k in range(len(img_names)):
                    if mat['names'][index][0][0] == img_names[k].split("/")[1][:-4]:
                        dst = os.path.join(ROOT, 'dg', str(i), str(f(mat['labels'][0,index])), mat['names'][index][0][0]+".jpg")
                        break
            cp_image(src, dst)
        print(f"finish domain {i}")
    
    print("haha")


if __name__ == "__main__":
    # EDGRotatedMNIST(0,0,{'env_density':10})
    gen_evol_circle()
    # OcularDisease_pre()
