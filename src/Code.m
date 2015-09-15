 %Loads text files into matrices
 Features = importdata('features.txt');
 Targets = importdata('targets.txt');
 Unknown = importdata('unknown.txt');
 
 HIDDENNEURONS = 8;
 
 %Generates random weight vectors
 inputWeights = (2.4/10 -(-2.4/10))*rand(10, HIDDENNEURONS) + (-2.4/10);
 outputWeights = (2.4/HIDDENNEURONS -(-2.4/HIDDENNEURONS))*rand(HIDDENNEURONS, 7) + (-2.4/HIDDENNEURONS);
 
 
 %The learning-phase starts here
 %Loops through all the rows of Features
 for i = 1: size(Features,1)
       input = Features(i,:);
       hiddenLayerNeurons = input * inputWeights;
       
 end