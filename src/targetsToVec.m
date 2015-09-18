clc;
clear all;

Targets = importdata('targets.txt'); 

TargetsVec = [];

for i=1: size(Targets,1)
    vec = transpose(ind2vec(Targets(i), 7));
    TargetsVec = [TargetsVec; vec];
end

TargetsVec;