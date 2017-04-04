function myTrain(trainSet,trainAnswers,epochs,minibatchSize,eta)
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
        updateMinibatch(minibatchSets(:,:,k),minibatchAnswerses(:,k),eta);
    end
    
end

    function updateMinibatch(minibatchSet,minibatchAnswers,eta)
        dzdx = vl_nnsoftmax(minibatchSet,minibatchAnswers,dzdy);
    end
end