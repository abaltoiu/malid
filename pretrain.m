function [D, W, A] = pretrain(Y, labels)

% INPUTS:
%   Y                       signals, (m x N) matrix
%   labels                  class labels vector
% OUTPUTS:
%   D                       pre-trained dictionary
%   W                       pre-trained classifier matrix
%   A                       pre-trained label consistency transformation


% Configuration
[m,N] = size(Y);
oc = 2;                            % dictionary overcompleteness factor
n = oc * m;                        % number of dictionary atoms
s = sqrt(n);                       % sparsity

labels_list = unique(labels);  
c = length(labels_list);           % number of classes

alpha = 4;                         % LC-KSVD alpha parameter
beta = 16;                         % LC-KSVD beta parameter
init_method = 2;                   % initialization for LC-KSVD, ie. trained 
                                   % atoms for each class and shared atoms                  
                                   

% Form label matrix
H = zeros(c, N);
for cls = 1: c                  
    H(cls, labels == labels_list(cls)) = 1;
end

% Form label consistency matrix
nc = floor(n/(c+1));                % evenly divide atoms per classes
nr = nc*(c+1);                      % total number of atoms (nr <= n)

Q = zeros(nr, N);
jj = 0;
for i = 1 : c                       % allocate atoms for each signal
    jc = find(H(i,:)==1);           % indices of signals from class i
    Q(jj+1:jj+nc,jc) = 1;
    jj = jj + nc;
end
Q(jj+1:jj+nc,:) = 1;                % shared dictionary


% Perform Label Consistent Dictionary Learning
[W, D, A] = clas_labelcon_dl(Y, H, Q, n, s, alpha, beta, init_method);

