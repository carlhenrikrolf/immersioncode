function [x] = getLayerNOutput(stim, N, net)
res = vl_simplenn(net, stim);
x = res(N).x;