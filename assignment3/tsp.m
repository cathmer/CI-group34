clc;
clear all;

% Read the file into a matrix
A = dlmread('hard maze.txt');
% Take out the top row of the matrix (which only represent the size --> see
% file). What remains, represents the maze
Maze = A(2:size(A,1), 1:size(A,2));

fileID = fopen('hard coordinates.txt'); 
% Read the coordinates into a vector
B = fscanf(fileID, '%d %s %d %s');
fclose(fileID);
% The starting location
startLoc = [B(1) B(3)];

% The file with the product locations
fileID = fopen('tsp products.txt'); 

% Read the product locations in a cell array
C = fscanf(fileID, '%d %s %d %s %d %s');
fclose(fileID);

% Number of products that need to be picked up
productCount = C(1);

% Matrix that will store each product number with its coordinates
products = zeros(3, productCount);

% Starting index to loop through the cell array that was collected from the
% file. 
index = 3;

% Store all the products in a matrix
for i=1: productCount
    % The productnumber
    productNumber = C(index);
    
    % The column the product is in 
    productColumn = C(index + 2) + 1;
    
    % The row the product is in
    productRow = C(index + 4) + 1;
    
    products(1, i) = productNumber;
    products(2, i) = productColumn;
    products(3, i) = productRow;
    
    % Starting index of next product
    index = index + 6;
end

% Number of iterations (i.e. Generations) that the algorithm will go
% through
ITERATIONS = 50;
% The number of people in the starting population
STARTING_POPULATION = 20;
% The number of people that are selected from the current population for
% creating offspring. These are the people with the best fitness.
TOP_SELECTION = 10;
% Fitness = Fitness_Qoutient / Routelength. Used for scaling.
FITNESS_QUOTIENT = 6400;
% The probability that two parents will use crossover to create offspring.
% If not, they will be cloned
CROSSOVER_PROB = 0.7;
% The probability that a child will be mutated.
MUTATION_PROB = 0.01;

% The population
Population = zeros(STARTING_POPULATION, productCount);

% Create random people
for i=1: STARTING_POPULATION
   % Create a array with all values from 1 to productCount present once, in
   % a unique order. This represent a route, where each number represents a
   % product number
   chromosome = randperm(productCount);
   
   % Add the chromosome (or person) to the population
   Population(i,:) = chromosome;
end

% A cell matrix which will be used to store calculated routes, so they won't
% have to be calculated again. 
Routes = cell(1000, 2);
% The number of routes that have been saved so far
savedRoutes = 0;

% NewPopulation represents the population after offspring has been created.
% For now, it equals the current Population.
NewPopulation = Population;

% Loop through the entire genetic algorithm for ITERATIONS times
for n=1: ITERATIONS
    % Assign the newpop to the population
    Population = NewPopulation;
    % A matrix which will store the fitness of each person
    PopulationFitness = zeros(STARTING_POPULATION, 1);
    
    disp('Iteration: ');
    disp(n);
    disp('Population: ');
    Population
    
    % Calculates the total fitness for each person, by calculating the
    % routes from product to product. The lower the total route length, the
    % higher the fitness
    for i=1: STARTING_POPULATION
        % Initialize fitness and total route length to 0
        fitness = 0;
        totalRouteLength = 0;
        
        % Starting and ending location converted to matrix coordinates
        startColumn = startLoc(1) + 1;
        startRow = startLoc(2) + 1;
        
        % Calculate the route from product to product, in the order the
        % current 'person' has them
        for j=1: productCount
            % The current person
            parent = Population(i, :);
            
            % A boolean which indicates whether the matrix with routes
            % already contains a specific route
            containsRoute = false;
            
            % Each person has to start at the starting location
            if (j == 1)
                endColumn = products(2, parent(j));
                endRow = products(3, parent(j));
            else
                % The previous product is where the route will start, the
                % current product is where the route will end
                startColumn = products(2, parent(j-1));
                startRow = products(3, parent(j-1));
                endColumn = products(2, parent(j));
                endRow = products(3, parent(j));
            end
            
            % Loop through all the savedRoutes to see if the currentroute
            % has been calculated before
            for k=1:savedRoutes
               % True if Routes contains this route
               if sum(Routes{k,1} == [startColumn startRow endColumn endRow]) == 4
                   containsRoute = true;
                   % Store the routelength of the route
                   routeLength = size(Routes{k,2},2);
                   break;
               end
            end
            
            % If true, the route doesn't need to be calculated again and
            % the fitness is immediately updated based on the routelength
            if containsRoute
                totalRouteLength = totalRouteLength + routeLength;
            % The route needs to be calculated: ANT ALGO
            else
                % An array which will contain the shortestRoute
                shortestRoute = zeros(1, size(Maze,1) * size(Maze,2));
                % The number of savedRoutes is increased by 1
                savedRoutes = savedRoutes + 1;
                % The starting and ending coordinates of this route are
                % stored in Routes
                Routes{savedRoutes, 1} = [startColumn startRow endColumn endRow];
                Routes{savedRoutes, 2} = shortestRoute;
                
                % Calculate the route h times, and take the shortest one
                for h=1:1
                    % Function which uses antoptimization to calculate the
                    % shortest route
                    route = antFunction(Maze, startColumn, startRow, endColumn, endRow);
                    % Check if this route is the shortest (and not
                    % non-existent)
                    if size(route,2) < size(shortestRoute,2) && size(route,2) > 0
                        % Add the currentRoute to Routes
                        Routes{savedRoutes,2} = route;
                        shortestRoute = route;
                    end
                end
                
                % calculate the routelength and use it to update the
                % total route length
                routeLength = size(shortestRoute, 2);
                totalRouteLength = totalRouteLength + routeLength;
            end
        end
        
        % Update the PopulationFitness with the fitness of the current
        % person
        PopulationFitness(i,1) = FITNESS_QUOTIENT / totalRouteLength;
    end
    
    disp('Population fitness: ');
    PopulationFitness
    
    % If n is not in its last iteration, generate offspring
    if n < ITERATIONS
        % Matrix which will contain the new population
        NewPopulation = zeros(STARTING_POPULATION, productCount);
        
        % Sorts unique values in ascending order
        [sortedValues,sortIndex] = unique(PopulationFitness(:)); 
        
        % If there are at least TOP_SELECTION unique values go in here
        if size(sortIndex,1) >= TOP_SELECTION
            % Get the index of the TOP_SELECTION highest values
            maxIndex = sortIndex(end - (TOP_SELECTION - 1) : end);
            
            % Store the highest fitness values in this matrix
            TopPopFitness = PopulationFitness(maxIndex);
            % Store the people corresponding to the highest fitness values
            % in this matrix
            TopPop = Population(maxIndex,:);
            % Calculate the sum of the highest fitness values
            totalFitness = sum(TopPopFitness);
            
            % Store the two persons with the highest fitness value in the
            % new population. This ensures that good routes are never
            % thrown away.
            NewPopulation(STARTING_POPULATION - 1,:) = TopPop(TOP_SELECTION - 1,:);
            NewPopulation(STARTING_POPULATION,:) = TopPop(TOP_SELECTION,:);
            
        % If there are less than TOP_SELECTION unique values (but more than
        % 1), use the entire population for generating offspring
        elseif size(sortIndex,1) > 1
            % Store the entire populationfitness in here
            TopPopFitness = PopulationFitness;
            % Store the entire population in here
            TopPop = Population;
            % Calculate the sum of all fitness values
            totalFitness = sum(TopPopFitness);
            
            % Store the highest two persons in the new population
            NewPopulation(STARTING_POPULATION - 1,:) = Population(sortIndex(end-1),:);
            NewPopulation(STARTING_POPULATION,:) = Population(sortIndex(end),:);
        else
            % All the people are the same, so stop iterating
            break;
        end
            
        % For loop to generate offspring. Each loop, two different parents
        % are randomly chosen (with different probabilities based on their
        % fitness), and those two parents generate two children. The loop
        % is done STARTING_POPULATION-2 / 2 times, because the highest two
        % parents are cloned into the new population.
        for h=1: (STARTING_POPULATION - 2) / 2
            % Matrix with 2 rows, each row representing one parent
            Parents = zeros(2, productCount);
            
            % Store the top of the population, with their fitnesses and
            % total fitness, in temporary variables
            TempTopPop = TopPop;
            TempTopPopFitness = TopPopFitness;
            tempTotalFitness = totalFitness;
    
            % This for loop selects two parents
            for j=1: 2
                % random number to determine which parent is chosen
                randomNumber = rand();
                
                % Loop through the temporary top population
                for i=1: size(TempTopPop,1)
                    % Take i-th person from population
                    if randomNumber < (sum(TempTopPopFitness(1:i) / tempTotalFitness))
                        % The selected parent
                        curParent = TempTopPop(i,:);
                        % Remove the selected parent so it cannot be chosen
                        % again
                        TempTopPop(i,:) = [];
                        % Update the totalfitness, subtracting the fitness
                        % of the selected parent
                        tempTotalFitness = tempTotalFitness - TempTopPopFitness(i);
                        % Remove the fitness of the selected parent
                        TempTopPopFitness(i) = [];
                        % Add the selected parent to the Parents matrix
                        Parents(j,:) = curParent;
                        break;
                    end
                end
            end
            
            % Random number to determine if the parents will use crossover
            % to generate offspring, or that they will be cloned into the
            % next generation.
            randCrossover = rand();
            
            % Use crossover to generate offspring
            if randCrossover < CROSSOVER_PROB
                % The starting index of the crossover
                crossoverStart = ceil(rand()*productCount);
                % The ending index of the crossover
                crossoverEnd = ceil((productCount - crossoverStart)*rand()) + crossoverStart; 
                % Arrays representing the children
                child1 = zeros(1, productCount);
                child2 = zeros(1, productCount);
                % Arrays representing the parents
                parent1 = Parents(1,:);
                parent2 = Parents(2,:);
                
                % add the elements from index crossoverstart to index
                % crossoverend from parent1 to child1. 
                child1(crossoverStart:crossoverEnd) = parent1(crossoverStart:crossoverEnd);
                
                % In this for loop, child1 is filled with the elements from
                % parent2 that it doesn't yet have, adding them in order.
                parentIndex = 1;
                for c=1:productCount
                    % If c lies outside the crossover, it means the child
                    % still needs an element on this position
                    if c < crossoverStart || c > crossoverEnd
                        addedEl = false;
                        
                        % Go through the elements of parent2 in order until
                        % one is found that isn't in the child yet
                        while ~addedEl
                            if (sum(child1 == parent2(parentIndex)) == 0)
                                child1(c) = parent2(parentIndex);
                                addedEl = true;
                            else
                                parentIndex = parentIndex + 1;
                            end
                        end
                    end
                end
                
                % Do the same as for child1, but with the parents switched
                child2(crossoverStart:crossoverEnd) = parent2(crossoverStart:crossoverEnd);
                parentIndex = 1;
                for c=1:productCount
                    if c < crossoverStart || c > crossoverEnd
                        addedEl = false;

                        while ~addedEl
                            if (sum(child2 == parent1(parentIndex)) == 0)
                                child2(c) = parent1(parentIndex);
                                addedEl = true;
                            else
                                parentIndex = parentIndex + 1;
                            end
                        end
                    end
                end
                
                % For loop to determine for each child if it will be
                % mutated
                for m=1:2
                    % Random number that determines if a child gets mutated
                    randMutation = rand();
            
                    if randMutation < MUTATION_PROB
                        % Get two random indices
                        rand1 = ceil(18*rand());
                        rand2 = ceil(18*rand());
                        
                        % Swap the numbers belonging to the random indices
                        if m == 1
                            child1([rand1 rand2]) = child1([rand2 rand1]);
                        else
                            child2([rand1 rand2]) = child2([rand2 rand1]);
                        end
                    end
                end
                
                % Add the children to the new population
                NewPopulation(h*2 - 1,:) = child1;
                NewPopulation(h*2,:) = child2;
            else
                % Clone the parents
                child1 = Parents(1,:);
                child2 = Parents(2,:);
                
                % do the same as above, mutate children (possibly)
                for m=1:2
                    randMutation = rand();
            
                    if randMutation < MUTATION_PROB
                        rand1 = ceil(18*rand());
                        rand2 = ceil(18*rand());

                        if m == 1
                            child1([rand1 rand2]) = child1([rand2 rand1]);
                        else
                            child2([rand1 rand2]) = child2([rand2 rand1]);
                        end
                    end
                end
                
                % Add the parents to the new population
                NewPopulation(h*2 - 1,:) = child1;
                NewPopulation(h*2,:) = child2;
            end
        end
    end
end
    
% Calculates the index of the person with the highest fitness
for i=1:size(PopulationFitness,1)
    if i == 1
        highestFitnessIndex = 1;
        highestFitness = PopulationFitness(i);
    elseif PopulationFitness(i) > highestFitness
        highestFitnessIndex = i;
        highestFitness = PopulationFitness(i);
    end
end

% Array which will store the complete route
completeRoute = [];
% The person with the highest fitness
parent = Population(highestFitnessIndex,:);

% Loop through the route from product to product for this person, to
% calculate the length of the entire route.
for j=1:productCount
    if (j == 1)
        startColumn = startLoc(1) + 1;
        startRow = startLoc(2) + 1;
        endColumn = products(2, parent(j));
        endRow = products(3, parent(j));
    else
        startColumn = products(2, parent(j-1));
        startRow = products(3, parent(j-1));
        endColumn = products(2, parent(j));
        endRow = products(3, parent(j));
    end
    
    for k=1:savedRoutes
        if sum(Routes{k,1} == [startColumn startRow endColumn endRow]) == 4
            completeRoute = [completeRoute, Routes{k,2}];
            break;
        end
    end
end

RESULT_FILE = 'tsp_results.txt';

dlmwrite(RESULT_FILE, []);
fileID = fopen(RESULT_FILE, 'wt'); 
results = [size(completeRoute, 2) + productCount, startLoc(1), startLoc(2)];
formatSpec = '%d;\n%d, %d;\n';
fprintf(fileID, formatSpec, results);

% Loop through the route from product to product for this person, to write
% it to file
for j=1:productCount
    if (j == 1)
        startColumn = startLoc(1) + 1;
        startRow = startLoc(2) + 1;
        endColumn = products(2, parent(j));
        endRow = products(3, parent(j));
    else
        startColumn = products(2, parent(j-1));
        startRow = products(3, parent(j-1));
        endColumn = products(2, parent(j));
        endRow = products(3, parent(j));
    end
    
    for k=1:savedRoutes
        if sum(Routes{k,1} == [startColumn startRow endColumn endRow]) == 4
            fprintf(fileID, '%d;', Routes{k,2});
            fprintf(fileID, 'take product #%d;', products(1, parent(j)));
            break;
        end
    end
end

fclose(fileID);
disp('Done!');



