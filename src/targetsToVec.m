%% Code to convert the Targets into a matrix which can be used by the MatLab Toolbox for Neural Networks

clc;
clear all;

% Import the targets
Targets = importdata('targets.txt'); 

TargetsVec = [];

% For each item of the targets, transpose it and convert it into a vector
% with 6 zero's and 1 one
for i=1: size(Targets,1)
    vec = transpose(ind2vec(Targets(i), 7));
    TargetsVec = [TargetsVec; vec];
end