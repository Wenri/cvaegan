function [ e ] = appProp( imin, g, w )
%APPPROP an implementation of "all-pairs appearance-space edit propagation"
%   im, input image in Lab color space and m-by-n-by-3 dimension
%   g, a vector user specified edits' parameters
%   w, a vector holds user specified strength of g, range in [0,1]
%   imout, output image, edited using AppProp algorithm
%   e, the propagated edit parameters
%   Author: Hao Lv
%   Email: lvhaoexp@163.com
%   Notice: this function is to be implemented.

if exist('U.mat','file')
%    load U;
    A = U(1:100,1:100);
else
    %get size
  [imsz.rows, imsz.cols, imsz.channels] = size(imin);

    %randomly sample 100 points in origin image
    sampleDim = 10;
    srows = sort(randperm(imsz.rows, sampleDim));
    scols = sort(randperm(imsz.cols, sampleDim));
    %create random selected image
    im = imin(srows, scols);

    %get feature map
    fvec_all = featVec(imin);
    fvec = fvec_all(srows, scols,:);

    %compute U
    delta_a_2 = 500;
    delta_s_2 = 10;
    A = zeros(sampleDim^2, sampleDim^2);
    B = zeros(  imsz.rows*imsz.cols - sampleDim^2, sampleDim^2 );

    fv_1 = zeros(1,3);
    fv_2 = zeros(1,3);
    for c=1:sampleDim^2
        sc = mod(c, sampleDim);
        sr = ceil(c/sampleDim);
        if sc == 0
            sc = sampleDim;
        end
        xi = [srows(sr),scols(sc)];
        fv_1 = squeeze(fvec(sr,sc,:));
        for r=1:sampleDim^2
            sc2 = mod(r, sampleDim);
            if sc2 == 0
                sc2 = sampleDim;
            end
            sr2 = ceil(r/sampleDim);
            fv_2 = squeeze(fvec(sr2,sc2,:));
            A(r,c) = exp(-norm(fv_1 - fv_2)/delta_a_2)*...
                     exp(-norm(xi - [srows(sr2), scols(sc2)])/delta_s_2);
        end

        rcnt = 0;
        elenum = imsz.rows*imsz.cols;
        cnt = 0;
        for r=1:elenum
            sc3 = mod(r, imsz.cols);
            sr3 = ceil(r/imsz.cols);
            if sc3 == 0
                sc3 = imsz.cols;
            end

            if(any(scols == sc3) && any(srows == sr3))
                cnt = cnt + 1;
                continue;
            end

            rcnt = rcnt + 1;
            fv_3 = squeeze(fvec_all(sr3,sc3,:));
            B(rcnt, c)= exp(-norm(fv_1 - fv_2)/delta_a_2)*...
                     exp(-norm(xi - [sr3, sc3])/delta_s_2); 
        end
    end
    U = [A;B];
    size(A)
    size(B')
%    save U.mat;
end
lamda = mean(w)
tic
one  = 0.5*lamda*U*(A\(U'*w'));
two = U*(A\(U'*ones(imsz.rows*imsz.cols,1)));
d = one + two;
toc
tic
D = spdiags(d(:),0,imsz.rows*imsz.cols, imsz.rows*imsz.cols);
itm1 = D\(U*(A\(U'*(w'.*g'))));
itm2 = D\(U*((-A+U'*(D\U))\(U'*(D\(U*(A\(U'*(w'.*g'))))))));

e = 0.5/lamda*(itm1-itm2);
toc
end
