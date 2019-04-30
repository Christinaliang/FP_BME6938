import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import cv2
import os,sys
import csv
import json
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def walk_dir(data_dir, file_types, filter=None):
    # file_types = ['.txt', '.kfb'], endswith
    # filter is function that return bool. True to include, False to not include
    # it can be labmda x:'unwanted string' not in x

    if not isinstance(file_types, list):
        assert 0, 'file_types must be list'
    #import pdb;pdb.set_trace()
    ff = os.walk(data_dir, followlinks=True)
    path_list = []
    for dirpath, dirnames, file_names in ff:
        for this_file_name in file_names:
            for this_type in file_types:
                if this_file_name.endswith(this_type):
                    this_path = os.path.join(dirpath, this_file_name)
                    if filter is not None:
                        if filter(this_path):
                            path_list.append( this_path  )
                    else:
                        path_list.append( this_path  )
                    break
    return path_list


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    file_types  = ['.png','.jpeg','.jpg']
    #import pdb;pdb.set_trace() 
    name = []
    captions = []
    ori = []
    img_list = []
    this_list = walk_dir(args.image_dir, file_types)
    with open('./data_bcidr/test_annotation.json', 'r') as f:
        json_file = json.load(f)
        #pprint(json_file)
        #import pdb;pdb.set_trace()
        #imglist = len(json_file)
        for img_name in json_file.keys():
            #print('Processing: ':img_name)
            img_list.append(img_name)
            #ori.append(json_file[img_name][0])


    for image in this_list:
        a = image.replace('./data_bcidr/test/','')
        image1 = os.path.splitext(a)[0]
        image = load_image(image, transform)
        image_tensor = image.to(device)
        
        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        name.append(image1)
        captions.append(sentence)
        #import pdb;pdb.set_trace()
        ori.append(json_file[image1]['caption'][0])
        # Print out the image and the generated caption
        #print (sentence)
        #image = Image.open(args.image)
        #plt.imshow(np.asarray(image))
    #import pdb;pdb.set_trace()
    #index_ = list( range(0, len(name) ) )
    #index_.sort(key=name.__getitem__)

    #csv_file = os.path.join(args.image_dir, 'test_result.csv')
    csv_file = 'test_result4.csv'
    print(len(name),len(captions),len(ori))
    with open(csv_file, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow( ['name', 'captions','ori','similarity'] ) 
        for ind in range(0, min(len(name)-1,len(captions)-1)):
            name1, caption1, ori1 =  name[ind], captions[ind].replace('<start> ',''), ori[ind]
            ratio = SequenceMatcher(None, caption1, ori1).ratio()
            #ratio = sentence_bleu(caption1, ori1)
            #print(ind)
            #print(name1)
            #print(caption)
            writer.writerow([name1, caption1,ori1,ratio])           


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data_bcidr/test', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models4/encoder-29-100.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models4/decoder-29-100.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data_bcidr/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=4096, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)