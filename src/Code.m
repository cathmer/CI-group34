%% This script learns to identify target classes based on 10 input variables
% Made by group 34 CI

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% NOTE: We used k-fold cross validation to find out what the best amount of
% hidden neurons is, so for this we needed a for loop that runs 10 times (in our case). 
% After we found out what the best amount of hidden neurons is, we didn't
% need it anymore, but the code is still in here, however the for loop is
% commented out
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% Clear the console and all variables before running
clc
clear all

%% Initiliaze variables
% Loads text files into matrices
Features = importdata('features.txt');  % A list of 7854 arrays of inputs
Targets = importdata('targets.txt');    % A list with outcomes corresponding to the inputs of Features
Unknown = importdata('unknown.txt');    % A list with inputs, with no outputs known

% splits the input matrix into 10 even pieces
FeaturesSplit = mat2cell(Features, [785,785,785,785,785,785,785,785,785,789], 10);
% Splits the targets into 10 even pieces
TargetsSplit = mat2cell(Targets, [785,785,785,785,785,785,785,785,785,789], 1);

%% K-fold cross validation

% Record the sum of MSE's for each kfold iteration
MSEsum = 0;

% Run through the entire training and testing process k times, while using
% a different testing set each time
%for kfold=1: 10

    kfold = 1;
    % Assign the featuressplit and targetssplit to new variables, so the
    % original variables remain unchanged
    FeaturesSplit1 = FeaturesSplit;
    TargetsSplit1 = TargetsSplit;
    
    % The last 10% of inputs is used as a test set and is converted back to a
    % regular matrix
    TestSet = cell2mat(FeaturesSplit1(kfold));
    % Remove the test set from the features
    FeaturesSplit1(kfold) = [];

    % The last 10% of targets is used as a test set and is converted back to a
    % regular column vector
    TestTargets = cell2mat(TargetsSplit1(kfold));
    % Remove the test targets from the features
    TargetsSplit1(kfold) = [];

    HIDDENNEURONS = 17;  % The number of hidden neurons we use
    alpha = 0.1;        % The learning curve we use

    % Generates a 10 by HIDDENNEURONS matrix with random weights values (between -2.4/10
    % and 2.4/10). The weight at position (i, j) corresponds to the connection
    % between the i-th input variable and the j-th hidden neuron.
    inputWeights = (2.4/10 -(-2.4/10))*rand(10, HIDDENNEURONS) + (-2.4/10);
    % Generates a row vector with random treshold values between -2.4/10 and
    % 2.4/10. The j-th value corresponds to the j-th hidden neuron.
    tresholdHiddenNeurons = (2.4/10 -(-2.4/10))*rand(1, HIDDENNEURONS) + (-2.4/10);

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
    EPOCHS = 20;
    % An array which will contain the Mean Squared Error of each epoch
    allErrors = [];
    allErrorsTraining = [];
    totalSuccessRate = 0;

    %% The learning-phase starts here
    % The number of times one epoch is executed
    for iterations = 1: EPOCHS
        errorsSquared = 0;  % This variable will hold the total of all errorsquared in one Epoch.

        % Generate a random integer between 1 and 9 to determine which part of
        % the Features will be the validation set
        random = randi([1, 9]);

        % Assign the featuressplit and targetssplit to new variables, so the
        % original variables remain unchanged and can be reused each iteration
        FeaturesSplit2 = FeaturesSplit;
        TargetsSplit2 = TargetsSplit;

        % Create a validation set and convert it back to a matrix
        ValidationSet = cell2mat(FeaturesSplit2(random));
        % Create validation targets
        ValidationTargets = cell2mat(TargetsSplit2(random));
        % Remove the validation set from the features and targets
        FeaturesSplit2(random) = [];
        TargetsSplit2(random) = [];
        % Create a trainingset and targets for this set
        TrainingSet = cell2mat(FeaturesSplit2);
        TrainingTargets = cell2mat(TargetsSplit2);

        %% Training
        % This loop runs through all the training inputs
        for n = 1: size(TrainingSet,1)
            input = TrainingSet(n,:);  % The current row of the input matrix

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
            desiredOutput = ind2vec(TrainingTargets(n), 7);

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
        
        % Calculates the MSE of the current epoch of the training set and
        % adds it to a vector
        MSE = 1/size(TrainingSet,1)*errorsSquared;
        allErrorsTraining(iterations) = MSE;
        errorsSquared = 0;

        %% Validation
        % variable which keeps track of the number of times the validation
        % set correctly predicts the output
        successCount = 0;
        % This loop runs through all the validation inputs
        for n = 1: size(ValidationSet,1)
            input = ValidationSet(n,:);  % The current row of the input matrix

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
            desiredOutput = ind2vec(ValidationTargets(n), 7);

            % Calculates a column vector with the error for each output neuron
            errors = desiredOutput - output;

            % Calculates the sum of all errors squared for the current output
            % and ads it to errorsSquared
            for e=1: size(errors)
                errorsSquared = errorsSquared + (errors(e)^2);
            end
            
            % Classification is a number, which represents the output our
            % network predicts for the given input
            classification = vec2ind(output);
            % If our classification equals the actual output, add 1 to the
            % successcount
            if classification == ValidationTargets(n)
                successCount = successCount + 1;
            end
        end
        
        %% Calculate if error of last Epoch is low enough to stop
        % Calculate the mean squared error of the last Epoch
        MSE = 1/size(ValidationSet,1)*errorsSquared;
        disp('iterations, MSE');
        disp(iterations);
        disp(MSE);

        % Add the MSE to a vector containing all MSE's for all epochs
        allErrors(iterations) = MSE;

        % Stop if the MSE of the last epoch is smaller than 0.1
        if MSE < 0.01
            break;
        end
        
        % Calculate the success rate of this validation, and add it to the
        % total
        successRate = successCount / size(ValidationSet,1);
        totalSuccessRate = totalSuccessRate + successRate;
    end
    
    % display the average success rate of the validation set
    disp('averagee success rate validation: ');
    validationSuccessRate = totalSuccessRate / iterations
    
    %% Plot all the MSE's
    plot(allErrors);
    plot(allErrorsTraining);

    %% Test the results against a test set, which has not been used previously
    % This loop runs through all the test inputs
    errorsSquared = 0;
    % variable to keep track of the count of successfull predictions
    successCount = 0;
    % Variable to store all the (desired) outputs in the 1 to N form
    desiredOutputsMatrix = [];
    outputsMatrix = [];
    
    for n = 1: size(TestSet,1)
        input = TestSet(n,:);  % The current row of the input matrix

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
        desiredOutput = ind2vec(TestTargets(n), 7);
        desiredOutputsMatrix = [desiredOutputsMatrix, desiredOutput];
        outputsMatrix = [outputsMatrix, ind2vec(vec2ind(output),7)];

        % Calculates a column vector with the error for each output neuron
        errors = desiredOutput - output;

        % Calculates the sum of all errors squared for the current output
        % and ads it to errorsSquared
        for e=1: size(errors)
            errorsSquared = errorsSquared + (errors(e)^2);
        end
        
        % Classification is a number, which represents the output our
        % network predicts for the given input
        classification = vec2ind(output);
        % If our classification equals the actual output, add 1 to the
        % successcount
        if classification == TestTargets(n)
            successCount = successCount + 1;
        end
    end
    
    % Displays the success rate of the testing set
    disp('succes rate testing: ');
    successRate = successCount / size(TestSet,1)
    
    % Calculate the MSE of the test set and display it
    MSE = 1/size(TestSet,1)*errorsSquared;
    
    % Add the MSE of the last test set to the total of MSE's
    MSEsum = MSEsum + MSE;
    
    plotconfusion(desiredOutputsMatrix, outputsMatrix);
%end

% The average of all MSE's throughout the 10 folds
MSEaverage = MSEsum/kfold;

% empty array which will store results
results = [];

for n = 1: size(Unknown,1)
    input = Unknown(n,:);  % The current row of the input matrix

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
    
    % This stores the current classification in an array with all results,
    % which will be written to a file afterwards
    results(n) = vec2ind(output);
end

% writes the classifications of all unknown inputs to a file
dlmwrite('unknown_classifications.txt', results);