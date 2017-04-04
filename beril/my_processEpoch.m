function [ net, state ] = my_processEpoch(net, state, params, mode)
% This also includes accumulateGradients
%
% This function is not made for running either. 

% initialize momentum = 0 for each corresponding edge in the graph
for i = 1:numel(net.layers)
    for j = 1:numel(net.layers{i}.weights)
        state.momentum{i}{j} = 0 ;
    end
end

profile on % important? for what?

if strcmp(mode, 'train') % For training
    dzdy = 1 ; % backpropagate
    evalMode = 'normal' ;
else
    dzdy = [] ; % do not backpropagate
    evalMode = 'test' ;
end
net.layers{end}.class = labels ; % extract the labels
res = vl_simplenn(net, im, dzdy, res, ...
    'accumulate', s ~= 1, ...
    'mode', evalMode, ...
    'conserveMemory', params.conserveMemory, ...
    'backPropDepth', params.backPropDepth, ...
    'sync', params.sync, ...
    'cudnn', params.cudnn, ...
    'parameterServer', parserv, ...
    'holdOn', s < params.numSubBatches) ; % get predictions for testing and
    % weight change for training

% accumulate errors
error = sum([error, [...
    sum(double(gather(res(end).x))) ;
    reshape(params.errorFunction(params, labels, res),[],1) ; ]],2) ;

% accumulate gradient
if strcmp(mode, 'train') % for training
    if ~isempty(parserv), parserv.sync() ; end
    %[net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv) ;
    
    for l=numel(net.layers):-1:1
        for j=numel(res(l).dzdw):-1:1
            parDer = res(l).dzdw{j}  ; % Here I get the gradient
            thisDecay = params.weightDecay * net.layers{l}.weightDecay(j) ; % maybe
            % a bit superfluous since its all 1
            thisLR = params.learningRate * net.layers{l}.learningRate(j) ; % The same
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.layers{l}.weights{j}) ; % for scalars this is just alpha*A + beta*B
                
                % Update momentum.
                state.momentum{l}{j} = vl_taccum(...
                    params.momentum, state.momentum{l}{j}, ...
                    -1, parDer) ;
                
                delta = state.momentum{l,j} ;
                
                % Update parameters.
                net.layers{l}.weights{j} = vl_taccum(...
                    1, net.layers{l}.weights{j}, ...
                    thisLR, delta) ; % Incorporate the new weights into the
                    % neural network
            end
        end
    end
end
end


