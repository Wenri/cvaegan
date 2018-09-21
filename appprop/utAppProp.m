%   unit test of appProp function
clear;
im = imread('../res/img/1.jpg', 'jpg');
cform = makecform('srgb2lab');
im_lab = applycform(im, cform);
[rows, cols, ~] = size(im);
g = ones(1, rows*cols);
g(1:100) = randperm(1000,100);
w = ones(1, rows*cols)/4;
[ e ] = appProp(im_lab, g, w);