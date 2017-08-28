**PredNet** implementation of PyTorch.

### Details
"Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning"(https://arxiv.org/abs/1605.08104)

The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005)

Original paper's [code](https://github.com/coxlab/prednet) is writen in Keras. Examples and project website can be found [here](https://coxlab.github.io/prednet/).


ConvLSTMCell is borrowed from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

### Training data
The preprocessed KITTI data is used which can be accessed using `downlaod_data.sh` from https://github.com/coxlab/prednet