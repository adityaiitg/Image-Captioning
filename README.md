# Image-Captioning using InceptionV3 and Beam Search

Using Flickr8k dataset since the size is 1GB. MS-COCO is 14GB!

Used <a href="https://keras.io/">Keras</a> with <a href="https://www.tensorflow.org/">Tensorflow</a> backend for the code. **InceptionV3** is used for extracting the features.

I am using Beam search with **k=3, 5, 7** and an Argmax search for predicting the captions of the images.

The loss value of **1.5987** has been achieved which gives good results. You can check out some examples below. The rest of the examples are in the jupyter notebook. You can run the Jupyter Notebook and try out your own examples. *unique.p* is a pickle file which contains all the unique words in the vocabulary. 

Everything is implemented in the Jupyter notebook which will hopefully make it easier to understand the code.
Interview_Bit_Revision
adityaiitg
/
C-Practice
adityaiitg
/
Image-Captioning
￼Show more
Working with a team?
GitHub is built for collaboration. Set up an organization to improve the way your team works together, and get access to more features.

Create an organization
Dashboard
￼
buckyroberts starred thenewboston-developers/thenewboston-node 2 days ago
thenewboston-developers/thenewboston-node
￼Star
Node for thenewboston digital currency network.

 Python 13 Updated Mar 10

￼
mukul54 starred POSTECH-CVLab/PyTorch-StudioGAN 10 days ago
POSTECH-CVLab/PyTorch-StudioGAN
￼Star
StudioGAN is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional ima…

 Python 1.5k Updated Mar 10

￼
konqr created a repository konqr/chess-imitation-learning 14 days ago
konqr/chess-imitation-learning
￼Star
Updated Feb 25

￼
vinaysomawat made vinaysomawat/angular public 18 days ago
vinaysomawat/angular
￼Star
This is Angular version of portfolio site.

 CSS 1 issue needs help Updated Feb 20

￼
konqr starred zhelyabuzhsky/stockfish 22 days ago
zhelyabuzhsky/stockfish
￼Star
Integrates the Stockfish chess engine with Python

 Python 72 Updated Mar 4

￼
buckyroberts created a repository thenewboston-developers/Activity-Reports 23 days ago
thenewboston-developers/Activity-Reports
￼Star
Weekly activity reports for teams and projects.

Updated Mar 10

￼
mukul54 starred justLars7D1/Reinforcement-Learning-Book 23 days ago
justLars7D1/Reinforcement-Learning-Book
￼Star
 TeX 35 Updated Feb 28

￼
buckyroberts created a repository thenewboston-developers/Research 28 days ago
thenewboston-developers/Research
￼Star
Research related material and documentation.

Updated Mar 9

￼
buckyroberts starred sno2/thenewboston-account-generator 29 days ago
sno2/thenewboston-account-generator
￼Star
Create accounts and server node options with ease!

 Vue 9 Updated Mar 1

￼
buckyroberts created a repository thenewboston-developers/Communications on 31 Jan
thenewboston-developers/Communications
￼Star
Updated Feb 15

￼
mukul54 starred emilwallner/Coloring-greyscale-images on 30 Jan
emilwallner/Coloring-greyscale-images
￼Star
Coloring black and white images with deep learning.

 Python 767 Updated Mar 9

￼
mukul54 starred haydengunraj/COVIDNet-CT on 23 Jan
haydengunraj/COVIDNet-CT
￼Star
COVID-Net Open Source Initiative - Models and Data for COVID-19 Detection in Chest CT

 Jupyter Notebook 54 Updated Mar 6

￼
parag-ag created a repository parag-ag/Keyboard-Automation on 22 Jan
parag-ag/Keyboard-Automation
￼Star
Updated Jan 22

￼
vinaysomawat created a repository vinaysomawat/bookrepo on 16 Jan
vinaysomawat/bookrepo
￼Star
Updated Jan 16

￼
buckyroberts starred richardkiss/pycoin on 16 Jan
richardkiss/pycoin
￼Star
Python-based Bitcoin and alt-coin utility library.

 Python 1.2k Updated Mar 10

￼
konqr starred Vinohith/Self_Driving_Car_specialization on 10 Jan
Vinohith/Self_Driving_Car_specialization
￼Star
Assignments and notes for the Self Driving Cars course offered by University of Toronto on Coursera

 Jupyter Notebook 188 Updated Mar 10

￼
mukul54 starred Hyperparticle/one-pixel-attack-keras on 7 Jan
Hyperparticle/one-pixel-attack-keras
￼Star
Keras implementation of "One pixel attack for fooling deep neural networks" using differential evolution on Cifar10 and ImageNet

 Jupyter Notebook 1.1k 3 issues need help Updated Mar 5

￼
buckyroberts starred lapstjup/animeccha on 6 Jan
lapstjup/animeccha
￼Star
A website to replay some of my favorite anime montages.

 TypeScript 26 Updated Mar 1

￼
mukul54 starred arshadshk/SAINT-pytorch on 29 Dec 2020
arshadshk/SAINT-pytorch
￼Star
SAINT PyTorch implementation

 Python 32 Updated Mar 4

￼
mukul54 starred PolyAI-LDN/conversational-datasets on 23 Dec 2020
PolyAI-LDN/conversational-datasets
￼Star
Large datasets for conversational AI

 Python 831 Updated Mar 10

￼
shashwatjolly made shashwatjolly/cvcj-bot public on 23 Dec 2020
shashwatjolly/cvcj-bot
￼Star
A fun bot for messenger group chats.

 JavaScript Updated Dec 25

￼
mukul54 starred jrieke/traingenerator on 16 Dec 2020
jrieke/traingenerator
￼Star
￼ A web app to generate template code for machine learning

 Python 941 Updated Mar 9

￼
mukul54 starred jettify/pytorch-optimizer on 15 Dec 2020
jettify/pytorch-optimizer
￼Star
torch-optimizer -- collection of optimizers for Pytorch

 Python 1.7k Updated Mar 10

 ProTip! The feed shows you events from people you follow and repositories you watch.
Subscribe
# Examples

!["first2"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/first2.jpg "first2")
!["second2"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/second2.jpg "second2")
!["third"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/third.jpg "third")
!["last1"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/last1.jpg "last1")

# Dependencies

* Keras 1.2.2
* Tensorflow 0.12.1
* tqdm
* numpy
* pandas
* matplotlib
* pickle
* PIL
* glob

# References
[1] Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan <a href="https://arxiv.org/abs/1411.4555">Show and Tell: A Neural Image Caption Generator</a>

[2] CS231n Winter 2016 Lesson 10 Recurrent Neural Networks, Image Captioning and LSTM <a href="https://youtu.be/cO0a0QYmFm8?t=32m25s">https://youtu.be/cO0a0QYmFm8?t=32m25s</a> 
