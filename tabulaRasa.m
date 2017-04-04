%%
imSize = [50,50];     % number of [rows, columns] for each samples, in pixels % 227,227
nSamples = 500;      % number of samples for each vernier side in each set
dataType = 'uint8';   % '(u)int8,16,32,64' ; 'logical' ; 'double/single' ; etc.
D = 1:10;             % various vernier offsets, in pixels
T = 1:5;              % various vernier thickness, in pixels
L = 5:12; 
[RCrowdedTrainSet, RCrowdedTestSet, LCrowdedTrainSet, LCrowdedTestSet] = createCrowdedSampleSets(imSize,nSamples,D,T,L,dataType);

a = 3;
for i = 1:a^2
    subplot(a,a,i)
    im = repmat(LCrowdedTestSet(:,:,i)*255,[1,1,3]);
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    imagesc(im_)
    %truesize
end
%truesize
%%
r = randi(min([size(LCrowdedTestSet,3)
    size(LCrowdedTrainSet,3)]));
subplot(1,2,1)
imagesc(LCrowdedTestSet(:,:,r))
subplot(1,2,2)
imagesc(LCrowdedTrainSet(:,:,r))
%truesize

%%
trainSet = 0;
testSet = 0;

epochs = 1;
minibatchSize = 1;
eta = 1;

function myTrain(trainSet,trainAnswers,epochs,batchSize,eta)
dzdy = 0;
for j = 1:epochs
    
    [n, m] = size(trainSet); % n = length(trainAnswers)
    
    shuffling = randperm(n);
    trainAnswers = trainAnswers(shuffling);
    trainSet = trainSet(shuffling);
    
    nMinibatches = floor(n/minibatchSize);
    if nMinibatches == 0
        nMinibatches = 1;
        minibatchSize = n;
    end
    minibatchAnswerses = zeros(minibatchSize,nMinibatches);
    minibatchSets = zeros(minibatchSize,m,nMinibatches);
    for k = 1:nMinibatches
    minibatchAnswerses(:,k) = trainAnswers(k:k + minibatchSize);
    minibatchSets(:,:,k) = trainSet(k:k + minibatchSize,:);
    end
    for k = 1:nMinibatches
        updateMinibatch(minibatchSets(:,:,k),minibatchAnswerses(:,k));
    end
    
end
end

function updateMinibatch(minibatchSet,minibatchAnswers)
    dzdx = vl_nnsoftmax(minibatchSet,minibatchAnswers,dzdy);
end

%%
classifier = struct('layers',[],'meta',[]);
classifier.layers = cell(1);
classifier.layers{1,1} = struct('name','prob','type','softmax','weights',cell(1,2),'precious',0);
%classifier.layers{1,1}.weights{1,1}% = [1,2;3,4];