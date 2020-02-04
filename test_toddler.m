% Copyright (c) 2018, 2019 Paul Irofti <paul@irofti.net>
% Copyright (c) 2019 Andra Baltoiu <andra.baltoiu@fmi.unibuc.ro>
%
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
%
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


%% Load dataset
% The sample dataset is generated at random and it is different from the
% two datasets referenced in the article.
% The dataset contains both a labeled set of signals (Y_pre, labels_pre) 
% for offline pre-training and an unlabeled set of signals (Y) for testing
% online TODDLeR.
clear; clc;
load('sample_db.mat');

%% Pretraining
% Run Label Consistent DL Classification (with shared dictionary) on
% labeled signals
[D, W, A] = pretrain(Y_pre, labels_pre);

%% Run TODDLeR
% Example 1. Use default parameters
%estimate = toddler(Y, D, W, A);

% Example 2. Use 'G2' update method with custom constraint parameters, 
% sparsity and forget factor
estimate = toddler(Y, D, W, A, 'Method','G2', 'Constraint',{4,16,8,8}, ...
    'Sparsity',3, 'Forget',0.9); 



