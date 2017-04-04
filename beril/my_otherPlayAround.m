clear ;
clc ;
run ../matconvnet/matlab/vl_setupnn ;

net = load('../nets/imagenet-caffe-alex.mat') ; % Load the network
net = vl_simplenn_tidy(net) ; % update and fill in values

im1 = imread('../images/peppers.png') ;
im1_ = single(im1) ; % note: 255 range
im1_ = imresize(im1_, net.meta.normalization.imageSize(1:2)) ;
im1_ = im1_ - net.meta.normalization.averageImage ;
im = im1_ ;

net.layers(end) = [] ;
net.layers(end+1) = struct('name','sm1',...
    'type','softmax',...
    'weights', [],...
    'size', [1,1,4096,1000], ...
    'pad', [0,0,0,0],...
    'stride', [1,1], ...
    'precious',0, ...
    'dilate', 1, ...
    'opts', []);
dzdy = 1 ;
res = vl_simplenn(net, im) ;
x = res(end).x ;


%res = 
dzdx = vl_nnsoftmax(x, dzdy) ;
