function processedSet = processImages(set, net)

len = size(set,3);
im = repmat(set(:,:,1)*255,[1,1,3]);
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;
shape = [size(im_),len];
processedSet = zeros(shape);
processedSet(:,:,:,1) = im_;
for i = 2:len
    im = repmat(set(:,:,i)*255,[1,1,3]);
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    processedSet(:,:,:,i) = im_;
end
    