%% This script learns to identify target classes based on 10 input variables
% Made by group 34 CI
% Clear the console and all variables before running
clc
clear all

%% Initiliaze variables
% Loads text files into matrices
Features = importdata('features.txt');  % A list of 7854 arrays of inputs
Targets = importdata('targets.txt');    % A list with outcomes corresponding to the inputs of Features
Unknown = importdata('unknown.txt');    % A lit with inputs, with no outputs known

HIDDENNEURONS = 8;  % The number of hidden neurons we use
alpha = 0.1;        % The learning curve we use

% Generates a 10 by HIDDENNEURONS matrix with random weights values (between -2.4/10
% and 2.4/10). The weight at position (i, j) corresponds to the connection
% between the i-th input variable and the j-th hidden neuron.
inputWeights = (2.4/10 -(-2.4/10))*rand(10, HIDDENNEURONS) + (-2.4/10);
% Generates a row vector with random treshold values between -2.4/10 and
% 2.4/10. The j-th value corresponds to the j-th hidden neuron.
tresholdHiddenNeurons = (2.4/10 -(-2.4/10))*rand(1, 8) + (-2.4/10);

% Generates a HIDDENNEURONS by 7 matrix with random weights values (between
% -2.4/HIDDENNEURONS and 2.4/HIDDENNEURONS). The weight at position (j, k)
% corresponds to the connection between the j-th hidden neuron and the k-th
% output neuron.
outputWeights = (2.4/HIDDENNEURONS -(-2.4/HIDDENNEURONS))*rand(HIDDENNEURONS, 7) + (-2.4/HIDDENNEURONS);
% Generates a row vector with random treshold values between
% -2.4/HIDDENNEURONS and 2.4/HIDDENNEURONS. The k-th value corresponds to
% the k-th output neuron.
tresholdOutputNeurons = (2.4/HIDDENNEURONS -(-2.4/HIDDENNEURONS))*rand(1, 7) + (-2.4/HIDDENNEURONS);

% The number of times the complete training set will be looped through
EPOCHS = 50;
% An array, initliazed with all zeros, which will contain the Mean Squared
% Error of each epoch
allErrors = zeros(EPOCHS, 1);

%% The learning-phase starts here
% The number of times one epoch is executed
for iterations = 1: EPOCHS
    errorsSquared = 0;  % This variable will hold the total of all errorsquared in one Epoch.
    
    % This loop runs through all the training inputs
    for n = 1: size(Features,1)
        input = Features(n,:);  % The current row of the input matrix
        
        % Calculates the values in the hidden neurons, before activation.
        hiddenLayerNeurons = (input * inputWeights) - tresholdHiddenNeurons; 
        % Calculates the output values of the hidden neurons.
        hiddenNeuronsOutput = sigmf(hiddenLayerNeurons, [1 0]);
        
        % Calculates the values in the output neurons, before
        % activation(i.e. sigmoid)
        outputNeurons = (hiddenNeuronsOutput*outputWeights) - tresholdOutputNeurons;
        % Calculates the output values of the output neurons and transposes
        % the vector to a column vector
        output = transpose(sigmf(outputNeurons, [1 0]));
        
        % Takes the current Target, and converts it to a vector with a 1 on
        % the index corresponding to the Target, and a zero on all other
        % spots.
        desiredOutput = ind2vec(Targets(n), 7);
        
        % Calculates a column vector with the error for each output neuron
        errors = desiredOutput - output;
        
        % Initialize a column vector of size HIDDENNEURONS, with all zeros
        errorsHiddenLayer = zeros(HIDDENNEURONS,1);
    
        % Loops over the outputs of all the output neurons
        for k=1: size(output)
            % Calculates the gradient for the current output neuron
            gradient = output(k)*(1-output(k))*errors(k);
            % Updates the treshold of the current neuron
            tresholdOutputNeurons(k) = tresholdOutputNeurons(k) + (alpha * -1 * gradient);
    
            % Loops over the column in the weights matrix (outputweights) corresponding to
            % the current neuron (column k)
            for j=1: size(outputWeights, 1)
                % The value of the current field in the matrix
                currentWeight = outputWeights(j,k);
                % Updates the weight of the field corresponding to (j,k) in
                % the matrix
                outputWeights(j, k) = currentWeight + (alpha * hiddenNeuronsOutput(j) * gradient);
                
                % The old weight times the gradient is added to the
                % errorvalue of the hidden neuron that corresponds to the 
                % current weight.
                errorsHiddenLayer(j) = errorsHiddenLayer(j) + currentWeight*gradient;
            end
        end
    
        % Transposes the output of the hidden neurons into a column vector
        hiddenOutputTransposed = transpose(hiddenNeuronsOutput);
        
        % Loops over all the outputs of the hidden neurons
        for h=1: size(hiddenOutputTransposed)
            % Calculates the gradient of the current hidden neuron
            gradient = hiddenOutputTransposed(h)*(1-hiddenOutputTransposed(h))*errorsHiddenLayer(h);
            % Updates the treshold of the current hidden neuron
            tresholdHiddenNeurons(h) = tresholdHiddenNeurons(h) + alpha* -1 * gradient;
            
            % Loops over the column in the weights matrix (inputweights) corresponding to
            % this neuron (column h)
            for i=1: size(inputWeights, 1)
                % Updates the weight for the field corresponding to (i, h)
                inputWeights(i, h) = inputWeights(i, h) + (alpha * input(i) * gradient);
            end
        end
        
        % Calculates the sum of all errors squared for the current output
        % and ads it to errorsSquared
        for e=1: size(errors)
            errorsSquared = errorsSquared + (errors(e)^2);
        end
    end
    
    % Calculate the mean squared error of the last Epoch
    MSE = 1/size(Features,1)*errorsSquared;
    disp(MSE);
    
    % Add the MSE to a vector containing all MSE's for all epochs
    allErrors(iterations) = MSE;
    
    % Stop if the MSE of the last epoch is smaller than 0.1
    if MSE < 0.1
        break;
    end
end

% Plot all the MSE's
plot(allErrors);