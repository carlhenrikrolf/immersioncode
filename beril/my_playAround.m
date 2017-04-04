%%% Playing around
% Just to understand MatConvNet
% 
% What I'm trying right now:
% batch size = 1
% # images = 1-3
% binary classification, in the end we want vernier or anti-vernier
%
% I've tried to reduce the code so that I only have the most essential
% parts left. There is probably much more to rebuild. Much about actually
% seeing how well it all is performing
%
% THIS CODE IS NOT MENT FOR RUNNING YET!!!!!

%% General start commands
clear ;
clc ;
run ../matconvnet/matlab/vl_setupnn ;

net = load('../nets/imagenet-caffe-alex.mat') ; % Load the network
net = vl_simplenn_tidy(net) ; % update and fill in values

%% Options
opts.backPropDepth = 1 ; % just go back one layer for training
opts.randomSeed = 1991 ; % just chose some number to make reproducible results
opts.numEpochs = 1 ; % start simle with just one epoch

%% Load images
% Already her I'm having some troubles. I don't see exactly how these imbd
% structs are built in cnn_train. I've tried running the cifar example,
% but I don't see imdb in the workspace.
im1 = imread('../images/peppers.png') ;
im1_ = single(im1) ; % note: 255 range
im1_ = imresize(im1_, net.meta.normalization.imageSize(1:2)) ;
im1_ = im1_ - net.meta.normalization.averageImage ;

im2 = imread('../images/twinpeaks.jpg') ;
im2_ = single(im2) ; % note: 255 range
im2_ = imresize(im2_, net.meta.normalization.imageSize(1:2)) ;
im2_ = im2_ - net.meta.normalization.averageImage ;

imdb.train = im1_ ;
imdb.test = im2_ ;

%% Neural network
net.layers(end) = [] ; % remove last layer

% Initialise learning rate and weight decay
% I don't exactly understand how to interprate this piece
% of code from cnn_train. There's a 1 for each weight in the NN,
% abd for the learningrate and weight decay respectively.
% So the thing I don't  get is more implementional:
% When I try to look at the structs I'm modifying here I can see
% no change from not having the following piece of code.
for i = numel(net.layers)
    J = numel(net.layers{i}.weights) ;
    net.layers{i}.learningRate = ones(1,J) ;
    net.layers{i}.weightDecay = ones(1,J) ;
end

% Binary error function
predictions = gather(res(end-1).x) ; % Take predictions from the second
% to last layer. Since I already got rid of the last layer should I
% have "res(end)" instead?
error = bsxfun(@times, predictions, labels) < 0 ; % compares the predictions
% with the the labels from the training set
err = sum(error(:)) ; % making the error into a scalar instead of a vector

% Running over different epochs
for epoch = 1:opt.numEpochs
    
    % seeding the random number generator
    rng(epoch + opts.randomSeed) ;
    
    % just renaming some parameters
    params = opts ;
    params.epoch = epoch ;
    params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ; % Take out
    % learning rate correspond to the weight numbered the same as the epoch
    % until all weights have had their learning rate picked out. Then only
    % the last one gets picked out. I don't really trust this
    % interpretation. What does it actually mean?
    params.train = opts.train(randperm(numel(opts.train))) ; % shuffle the
    % training set
    params.val = opts.val(randperm(numel(opts.val))) ; % shuffle the
    % validation set
    params.imdb = imdb ;
    params.getBatch = getBatch ; % This getBatch thing is still
    % quite incomprehensible to me. I understand that it is some kind of
    % object containing tensors: height * width * # channels * # images, for
    % the images and lables: label patches (terminology?) for the y axis *
    % the same for x * 1 (why only 1?) * number of images. But I'm still
    % not so comfortable working with these struct things in matlab.
    
    % For this function, see a separate file.
    [net, state] = processEpoch(net, state, params, 'train') ;
    [net, state] = processEpoch(net, state, params, 'val') ;
    
end

%% Visualise
vl_simplenn_display(net) ;