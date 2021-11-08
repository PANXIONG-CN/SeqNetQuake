# A deep-learning method based on a sequence-based classification neural network for pre-earthquake perturbation identification (SeqNetQuake)
The SeqNetQuake code should be compiled with keras and Tensorflow


"""

@auther: Dr. Pan Xiong 

@email: xiongpan@ief.ac.cn 

"""


Usage
=======

step1: Install Keras with TensorFlow backend

step2: Download the file from https://drive.google.com/file/d/1krvqouo00Dh5VR_svamr8KoSZgy7iXTf/view?usp=sharing and unzip it to the INPUT directory

Testing
=======

Open an IPython shell ::

    >>> python RunModel.py -mn CNNBiLSTM -tr CSES_train -te CSES_test
