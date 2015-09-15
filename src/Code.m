clc
clear all

%Loads text files into matrices
Features = importdata('features.txt');
Targets = importdata('targets.txt');
Unknown = importdata('unknown.txt');

HIDDENNEURONS = 8;
alpha = 0.1;
errorsSquared = 0;

%Generates random weight vectors
inputWeights = (2.4/10 -(-2.4/10))*rand(10, HIDDENNEURONS) + (-2.4/10);
tresholdHiddenNeurons = (2.4/10 -(-2.4/10))*rand(1, 8) + (-2.4/10);

outputWeights = (2.4/HIDDENNEURONS -(-2.4/HIDDENNEURONS))*rand(HIDDENNEURONS, 7) + (-2.4/HIDDENNEURONS);
tresholdOutputNeurons = (2.4/HIDDENNEURONS -(-2.4/HIDDENNEURONS))*rand(1, 7) + (-2.4/HIDDENNEURONS);

allErrors = [];

%% The learning-phase starts here
%Loop through all the rows of the Features matrix, which is the input
for iterations = 1: 100
    for n = 1: size(Features,1)
        input = Features(n,:);  % The current row of the matrix, which is the input
        % Calculates the values in the 
        hiddenLayerNeurons = (input * inputWeights) - tresholdHiddenNeurons;  
        hiddenNeuronsOutput = sigmf(hiddenLayerNeurons, [1 0]);

        outputNeurons = (hiddenNeuronsOutput*outputWeights) - tresholdOutputNeurons;
        output = transpose(sigmf(outputNeurons, [1 0]));
        desiredOutput = ind2vec(Targets(n), 7);
        errors = desiredOutput - output;
        errorsHiddenLayer = zeros(HIDDENNEURONS,1);

        for k=1: size(output)
            gradient = output(k)*(1-output(k))*errors(k);
            tresholdOutputNeurons(k) = tresholdOutputNeurons(k) + (alpha * -1 * gradient);

            for j=1: size(outputWeights, 1)
                currentWeight = outputWeights(j,k);
                outputWeights(j, k) = currentWeight + (alpha * hiddenNeuronsOutput(j) * gradient);
                errorsHiddenLayer(j) = errorsHiddenLayer(j) + currentWeight*gradient;
            end
        end

        hiddenOutputTransposed = transpose(hiddenNeuronsOutput);

        for h=1: size(hiddenOutputTransposed)
            gradient = hiddenOutputTransposed(h)*(1-hiddenOutputTransposed(h))*errorsHiddenLayer(h);
            tresholdHiddenNeurons(h) = tresholdHiddenNeurons(h) + alpha* -1 * gradient;

            for i=1: size(inputWeights, 1)
                currentWeight = inputWeights(i, h);
                inputWeights(i, h) = currentWeight + (alpha * input(i) * gradient);
            end
        end

        for e=1: size(errors)
            errorsSquared = errorsSquared + (errors(e)^2);
        end
    end
    
    allErrors = [allErrors; errorsSquared];
    
    if 1/size(Features,1)*errorsSquared < 0.1
        break;
    end
end

plot(allErrors);