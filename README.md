
There are two folders inside the whole profile, and data is saved in data_bcidr folder, with the traing and testing images. the image set is partial, and I only leave 170 samle images in the training folder. As model and data folder are big, so I upload them into dropbox and can be downloaded by the below url:
https://www.dropbox.com/sh/fv2tsxybsa06in8/AACr1IYJYrC92Ss9kURELiysa?dl=0

#### 1. Preprocessing

This step is used for installing the required packages.
```bash
$ pip install -r requirements.txt   
```

#### 2. Building vocab

This step has been done for generating vocab , and the vocab.pkl is saved inside the data folder.
```bash
$ python build_vocab.py   
```

#### 3. Train the model
Can just run train.py for training setp. The default path is saved in args function, and if needed they can be changed.
```bash
$ python train.py    
```

#### 4. Test the model 
It will run the testing process for the images in the testing foler. The result will be saved as one csv file in the root folder.

```bash
$ python test.py
```
