function [prediction] = getOneClassifierPrediction(stim,net,N,classifier)

% prepare stimulus to go through DNN
im = repmat(stim(:,:,1)*255,[1,1,3]);
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% process with DNN
res = vl_simplenn(net, im_);
DNNout = res(N).x(:);

% classify
prediction = classifier(DNNout);