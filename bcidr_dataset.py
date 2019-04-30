
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import json
from build_vocab import Vocabulary
#from .dataset_utils import Vocabulary

class BCIDRDatasetUnary(data.Dataset):
    """CUB Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, which_set, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.img_root = os.path.join(root, 'Images')
        self.ann = json.load(open(os.path.join(root, '{}_annotation.json'.format(which_set)),'r'))

        self.vocab = vocab
        self.transform = transform
        self.img_list = list(self.ann.keys())
        # transfer categories id to labels
        self.num_ann = 1
        self.num_cats = 4
        self.ids = [a for a in range(len(self.ann) * self.num_ann)]
        self.cat2label = {}

        print('\t {} samples from {} set'.format(len(self.ids), which_set ))

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
        img_id = index // self.num_ann
        text_id = np.random.randint(0,4,1)[0]
        #text_id = 0
        img_name = self.img_list[img_id]
        caption = self.ann[img_name]['caption'][text_id]
        #img_labels = np.array([self.ann[img_name]['label']], np.int64)

        image = Image.open(os.path.join(self.img_root, img_name+'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
      
        targets = [c.lstrip(' ') for c in caption.split('.') if len(c) > 0] 
        #print(len(targets))
        #assert(len(targets) == 5)
        
        # Convert caption (string) to word ids.
        '''
        captions = []
        for target in targets: 
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            # caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)
        print(captions)
        '''
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target #, torch.from_numpy(img_labels)

    def __len__(self):
        return len(self.ids) 

label2word = {
    0: 'normal',
    1: 'low grade',
    2: 'high grade',
    3: 'insufficient information'
}

class BCIDRDataset(data.Dataset):
    """CUB Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, which_set, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.img_root = os.path.join(root, 'Images')
        self.ann = json.load(open(os.path.join(root, '{}_annotation.json'.format(which_set)),'r'))

        self.vocab = vocab
        self.transform = transform
        self.img_list = list(self.ann.keys())
        # transfer categories id to labels
        self.num_ann = 5

        self.num_cats = 4
        self.ids = [a for a in range(len(self.ann) * self.num_ann)]
        self.cat2label = {}

        print('\t {} samples from {} set'.format(len(self.ids), which_set ))

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
        img_id = index // self.num_ann
        text_id = index % self.num_ann 

        img_name = self.img_list[img_id]
        caption = self.ann[img_name]['caption'][text_id]
        #img_labels = np.array([self.ann[img_name]['label']], np.int64)

        image = Image.open(os.path.join(self.img_root, img_name+'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        targets = [c.lstrip(' ') for c in caption.split('.') if len(c) > 0] + [label2word[img_labels[0]]]
        #assert(len(targets) == 6)
        
        # Convert caption (string) to word ids.
        captions = []
        for target in targets: 
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            # caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)

        return image, torch.FloatTensor(captions) #torch.from_numpy(img_labels),

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    #import pdb;pdb.set_trace()

    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    #print(lengths)
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    '''
    coco = CocoDataset(root=root,
                    json=json,
                    vocab=vocab,
                    transform=transform)
    '''
    train_dataset_img = BCIDRDatasetUnary(root = '/home/yunliang/Data/YunDisk/data_bcidr/', 
                                    which_set='train', 
                                    vocab=vocab,
                                    transform=transform,
                                    )
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    #data_loader = DataLoader(train_dataset_img, batch_size=args.batch_size,
    #            shuffle=True, num_workers=args.num_workers,drop_last=True)


    data_loader = torch.utils.data.DataLoader(dataset=train_dataset_img, 
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn)
    return data_loader