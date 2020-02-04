
Keypoints Detection
==========================

This project aims at detecting keypoints using 

- FRCNN: [A Global-Local Emebdding Module for Fashion Landmark Detection](https://arxiv.org/pdf/1908.10548.pdf)
- Implementation: [Keras](https://github.com/benybrahim/keras_fashion_gle)

# Dependencies

- Python3 
- Keras 2.2.4
- Tensorflow 2.0 
- Opencv 4

# Installation 

    $ pip install --upgrade -r requirements.txt


# Download datasets and models

- **Train:** 
  - **images:** to come
  - **train.txt:** to come
  - **test.txt:** to come

- **Models:**
  - **model.hdf5:** to come


# Getting start

1. Put **images**, **train.csv**, **test.csv** and **test** data in `./data` folder.

2. Put **model.hdf5** in `./models` folder.

3. Change **'model_path'** and **'labels_path'** in `./utils/config.py`:

    - **model.hdf5** and **labels.txt** for board cropping
    - **model2.hdf5** and **labels2.txt** for production cropping

### Train model

    $ python train.py 
                       
### Use model

    $ python detect_kp.py --img {IMG_PATH} 


# The directory structure
------------
```
├── README.md        
│
├── .gitignore        
│
├── requirements.txt 
│
├── models   
│   └── model.h5 
│
├── data
│   ├── images        
│   ├── train.csv    
│   └── val.csv    
│
├── source                
│   ├── config.py      
│   │                    
│   ├── train.py      
│   │  
│   ├── train.py      
│   │
│   └── utils.csv 
│       ├── neural_net.py
│       ├── generator.py
│       └── metrics.py
```

