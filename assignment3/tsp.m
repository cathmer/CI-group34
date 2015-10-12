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

fileID = fopen('tsp products.txt'); 

C = fscanf(fileID, '%d %s %d %s %d %s');
fclose(fileID);

productCount = C(1);

products = zeros(3, 18);
index = 3;

for i=1: productCount
    productNumber = C(index);
    productColumn = C(index + 2) + 1;
    productRow = C(index + 4) + 1;
    
    products(1, i) = productNumber;
    products(2, i) = productColumn;
    products(3, i) = productRow;
    
    index = index + 6;
end

STARTING_POPULATION = 2;
FITNESS_QUOTIENT = 6400;

Population = zeros(STARTING_POPULATION, productCount);

for i=1: STARTING_POPULATION
   chromosome = randperm(productCount);
   
   Population(i,:) = chromosome;
end

Routes = cell(1000, 2);
savedRoutes = 0;

PopulationFitness = zeros(STARTING_POPULATION, 1);

% Fitness
for i=1: STARTING_POPULATION
    fitness = 0;
    % Starting and ending location converted to matrix coordinates
    startColumn = startLoc(1) + 1;
    startRow = startLoc(2) + 1;
    
    for j=1: productCount
        parent = Population(i, :);
        
        containsRoute = false;
        
        if (j == 1)
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
               containsRoute = true;
               routeLength = size(Routes{k,2},2);
               break;
           end
        end
        
        if containsRoute
            fitness = fitness + (FITNESS_QUOTIENT / routeLength);
        else
            shortestRoute = zeros(1, size(Maze,1) * size(Maze,2));
            savedRoutes = savedRoutes + 1;
            Routes{savedRoutes, 1} = [startColumn startRow endColumn endRow];
            Routes{savedRoutes, 2} = shortestRoute;
            
            for h=1:1
                route = antFunction(Maze, startColumn, startRow, endColumn, endRow);
                if size(route,2) < size(shortestRoute,2) && size(route,2) > 0
                    Routes{savedRoutes,2} = route;
                    shortestRoute = route;
                end
            end
            
            routeLength = size(shortestRoute, 2);
            fitness = fitness + (FITNESS_QUOTIENT / routeLength);
        end
        
        disp('Route number: ');
        disp(j);
        disp('Person number: ');
        disp(i);
    end
    
    PopulationFitness(i,1) = fitness;
    
end

for i=1:size(PopulationFitness,1)
    if i == 1
        highestFitnessIndex = 1;
        highestFitness = PopulationFitness(i);
    elseif PopulationFitness(i) > highestFitness
        highestFitnessIndex = i;
        highestFitness = PopulationFitness(i);
    end
end

completeRoute = [];
parent = Population(highestFitnessIndex,:);

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
            disp('K: ');
            k
            disp('Route from to: ');
            Routes{k,1}
            completeRoute = [completeRoute, Routes{k,2}];
            break;
        end
    end
end

RESULT_FILE = 'product_results.txt';

dlmwrite(RESULT_FILE, []);
fileID = fopen(RESULT_FILE, 'wt'); 
results = [size(completeRoute, 2), startLoc(1), startLoc(2)];
formatSpec = '%d;\n%d, %d;\n';
fprintf(fileID, formatSpec, results);
fprintf(fileID, '%d;', completeRoute);
fclose(fileID);

disp('Done!');



