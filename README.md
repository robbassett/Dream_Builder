# DREAM BUILDER
After playing around with the deep dream code provided by Tensorflow and utilizing the InceptionV3 network with weights trained on ImageNet, I wanted to see if I could use my own custom CNN trained on some other dataset. The main trick is that the input layer typically has a specified input shape while a deep dream generator needs to accept images of any size. Here I've simply written a function that strips the first layer off the input network and replaces it with a nearly identical layer having input shape = (None, None, 3). Currently I'm also removing any Flatteninng layers as well as any fully connected layers following this... though this may not be necessary? 

some code borrowed from
https://www.tensorflow.org/tutorials/generative/deepdream

Check "test_networks.py" for usage examples

**Notes: currently only works with .jpg input and networks trained on 3-channel images
