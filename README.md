# VGG16
The project expands on the work done in <a href="https://github.com/LukaszObara/LeNet5">LeNet5</a>. In this project we built and test an implementation of VGG16, a convolutional network described by the Visual Geometry Group in <em>Very Deep convolutional netowrks for large-scale image recognition</em>. As with <a href="https://github.com/LukaszObara/LeNet5">LeNet5</a> the code will be written in `python` using `theano`. We also make a few modification to the network to improve performance. 

### Architecture
The base network consist of a series of convolutional layers, pooling layers, and fully connected layers. The network that we built will also include a series of batch normalization layers between before each pooling layer, and a drop-out layer between the fully connected layers. We use an `Exponential Linear Unit (ELU)` as our activation function, max-pooling, and we train our network on a binary classification problem and use `ADAM` as our backpropagation methed. The diagram below illustrates the architecture:

<p align="center"><img src="https://github.com/LukaszObara/VGG16-Theano/blob/master/images/Archit.png" height="750"></p>

# References
<ol>
<li>Bengio Yoshua, Glorat Xavier, <em>Understanding the difficulty of training deep feedforward neural networks</em>, AISTATS, pages 249–256, 2010</li>
<li>Clevert Djork-Arne, Unterthiner Thomas, Hochreiter Sepp, <em>Fast And Accurate Deep Network Learning by Exponential Linear Units (ELU)</em>, ICLR 2016, https://arxiv.org/abs/1511.07289</li>
<li>Goodfellow Ian, Bengio Yoshua, Courville Aaron, <em>Deep Learning</em>, MIT Press, 2016, http://www.deeplearningbook.org</li>
<li>He Kaiming et al., <em>Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</em>, ICCV, 2015, pp. 1026-1034</li>
<li>Ioffe Sergey, Szegedy Christian, <em>Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</em>, JMLR, 2015 https://arxiv.org/abs/1502.03167</li>
<li>Kingma Diederik P., Lei Ba Jimmy, <em>ADAM: A Method For Stochastic Optimization</em>, ICLR, 2015, https://arxiv.org/pdf/1412.6980.pdf
</li>
<li>Krizhevsky Alex, Sutskever Ilya, Hinton Geoffrey E., <em>ImageNet Classification with Deep Convolutional Neural Networks</em>, NIPS, 2012, https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf</li>
<li>LeCun Yann et al., <em>Gradient-Based Learning Applied to Document Recognition</em>, PROC. OF THE IEEE., Nov 1998</li>
<li>Simonyan Karen, Zisserman Andrew, <em>Very Deep convolutional netowrks for large-scale image recognition</em>, ICLR 2015, https://arxiv.org/pdf/1409.1556.pdf</li>
<li>Srivastava Nitish et al., <em>Dropout: A Simple Way to Prevent Neural Networks from Overfitting</em>, JMLR, 2014, 
</ol>

## Notes
The network was tested on AWS using the `Dogs vs. Cats` data available on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition). Due to financial limitation the network was test for only 51 epochs. 
