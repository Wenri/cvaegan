%unit test for featvec function
clear;
im = imread('../res/img/1.jpg', 'jpg');
cform = makecform('srgb2lab');
im_lab = applycform(im, cform);
tic
fvec = featVec(im_lab);
toc

% ft = ones(3,3)/9;
% tic
% avg_img1 = imfilter(im_lab(:,:,1), ft);
% avg_img2 = imfilter(im_lab(:,:,2), ft);
% avg_img3 = imfilter(im_lab(:,:,3), ft);
% toc
