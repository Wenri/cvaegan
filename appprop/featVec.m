function [ fvecs ] = featVec( img )
%FEATVEC used to extract feature vector
%   [ fvecs ] = featVec( img )
%   img is image in Lab color space, m-by-n-by-3 in Lab order
%   fvecs is m-by-n-by-dim matrix which contains feature vector for each
%   pixel, here dim is 5 by default
%   Author : lvhao
%   Email:   lvhaoexp@163.com
[rows, cols, ~] = size(img);
paddedImg = zeros(rows + 2, cols + 2, 3, 'like', img);
paddedImg(2:rows+1, 2:cols+1,:) = img;
fdim = 9;
wsz = 9;
fvecs = zeros(rows, cols, fdim);
ft = ones(3,3)/wsz;
avg_img1 = imfilter(img(:,:,1), ft);
avg_img2 = imfilter(img(:,:,2), ft);
avg_img3 = imfilter(img(:,:,3), ft);
for r = 2:rows
    for c = 2:cols
        wnd = paddedImg(r-1:r+1,c-1:c+1,:);
        pwnd1 = power(wnd(:,:,1) - avg_img1(r-1,c-1), 2);
        pwnd2 = power(wnd(:,:,2) - avg_img2(r-1,c-1), 2);
        pwnd3 = power(wnd(:,:,3) - avg_img3(r-1,c-1), 2);
        fvecs(r,c,7:9) = [sqrt(sum(pwnd1(:))/wsz), sqrt(sum(pwnd2(:))/wsz), sqrt(sum(pwnd3(:))/wsz)];
    end
end
fvecs(:,:,1:3) = img;
fvecs(:,:,4) = avg_img1;
fvecs(:,:,5) = avg_img2;
fvecs(:,:,6) = avg_img3;
end

