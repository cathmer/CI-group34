clc;
clear all;

% File to which the results will be written
RESULT_FILE = 'easy_result.txt';

% Read the file into a matrix
A = dlmread('easy maze.txt');
Fieldsize = A(1,1)+ A(1,2);
% Take out the top row of the matrix (which only represent the size --> see
% file). What remains, represents the maze
Maze = A(2:size(A,1), 1:size(A,2));

fileID = fopen('easy coordinates.txt'); 
% Read the coordinates into a vector
C = fscanf(fileID, '%d %s %d %s');
fclose(fileID);
% The starting location
startLoc = [C(1) C(3)];
% The ending location
endLoc = [C(5) C(7)];

% Maximum # of iterations before the algorithm terminates
MAX_ITERATIONS = 500;
% # of ants per iteration
ANTS_PER_ITERATION = 5;
% The guess of the length of the route is approximated though the sum of the 
% length of the two sides a and b of the matrix.
% It is a multiplier for the amounts of pheromomens that are dropped.
% It is updated every iteration with the length of the so far shortest way.
PHEROMONES_DROPPED = 40;
% Fieldsize
% The rate at which pheromones evaporate
EVAPORATION_RATE = 0.01;
% The route length for which the algorithm is satisfied
CONVERGENCE_CRITERION = 30;
% Ants making it alive: antsalive
antsstuck = 0;
Been2 = 0;
Been4 = 0;
Been3 = 0;
Been1 = 0;
Been0 = 0;



% Vector which will store the actions of the shortest route
shortestRoute = zeros(1, size(Maze,1) * size(Maze,2));

% Starting and ending location converted to matrix coordinates
startColumn = startLoc(1) + 1;
startRow = startLoc(2) + 1;
endColumn = endLoc(1) + 1;
endRow = endLoc(2) + 1;

% Matrices which contain pheromone values for each square in one particular
% direction
%PherNorth = [zeros(1,size(Maze,2)); ones(size(Maze,1)-1, size(Maze,2))];
PherNorth = ones(size(Maze,1), size(Maze,2));
PherEast = ones(size(Maze,1), size(Maze,2));
PherSouth = ones(size(Maze,1), size(Maze,2));
PherWest = ones(size(Maze,1), size(Maze,2));

for i=1:MAX_ITERATIONS
    % These are temporary pheromone matrices for each direction. These will
    % contain the added pheromones of all the ants for this cycle
    TempPherNorth = zeros(size(Maze,1), size(Maze,2));
    TempPherEast = zeros(size(Maze,1), size(Maze,2));
    TempPherSouth = zeros(size(Maze,1), size(Maze,2));
    TempPherWest = zeros(size(Maze,1), size(Maze,2));
    
    for j=1: ANTS_PER_ITERATION
        hasLoop = 5;
        
        % The maze for the current ant. This tempmaze will be update -->
        % When an ant has visited a location, the value changes to 0.
        TempMaze = Maze;
        % The current location of an ant, which always starts at the same
        % location
        currentColumn = startColumn;
        currentRow = startRow;
        
        actions = [];
        
        % While loop stops when ant is at the destination
        while ~(currentColumn == endColumn && currentRow == endRow)
            
            % Set the current location to -1, to indicate the ant has visited it
            TempMaze(currentRow, currentColumn) = -1;
            
            % counts, at how many of neighbour cells ant has been before
            been = 0;
            
            % variable will be set 0,5 if ant steps back on a cell it
            % visited before. This way, the pheromone of this loop will be
            % decreased.
            ownEnd = 1;
            
            % Determine the probabilities for each adjacent square
            % North
            if (currentRow == 1)
                north = 0;
            else
                north = PherNorth(currentRow, currentColumn) * TempMaze(currentRow - 1, currentColumn);
                % If the destination candidate has been visited before, its
                % value is -1. The been counter gets to know, and the
                % delta-pheromone is set to 0.
                if (north<0)
                        been = been +1;
                end
            end
            
            %South
            if (currentRow == size(Maze,1))
                south = 0;
            else
                south = PherSouth(currentRow, currentColumn) * TempMaze(currentRow + 1, currentColumn);
                    if (south<0)
                    been = been +1;
                    end
            end
            
            % West
            if (currentColumn == 1)
                west = 0;
            else
                west = PherWest(currentRow, currentColumn) * TempMaze(currentRow, currentColumn -1);
                if (west<0)
                    been = been +1;
                end
            end
            
            % East
            if (currentColumn == size(Maze,2))
                east = 0;
            else
                east = PherEast(currentRow, currentColumn) * TempMaze(currentRow, currentColumn + 1);
                if (east<0)
                    been = been +1;
                end
            end
            
            total = north + south + west + east;

            % total is zero or smaller if ant got stuck. if it got stuck because it
            % has wall on three sides (deadend), it shall block the spot it is on
            % now, to close this dead-end. 
            % 
            % if ant got stuck because it has visited TWO of the nieghbour
            % cells (ONE where it just came from, and ONE previously on the
            % current path, it will be allowed to live on, but the
            % pheromone emission will be cut half
            if (total < 0 && been==2 && hasLoop>0) 
                ownEnd = 0.5;
                hasLoop = hasLoop - 1;
                if (east<0)
                    east = east * -1;
                end 
                if (west<0)
                    west = west * -1;
                end
                if (south<0)
                    south = south * -1;
                end
                if (north<0)
                    north = north * -1;
                end  
            end

            % If total is <= 0 but been is unequal 2; the direction values
            % of the previously visitd cells have to be set zero again.
            if(east<0)
                east = 0;
            end
            if (north<0)
                north = 0;
            end
            if (west<0)
                west = 0;
            end
            if (south<0)
                south = 0;
            end
                
            % the total is new calculated to exclude the been=2 case from
            % the kill-routine;
            total = north + south + west + east;
            
            if (total == 0)
                antsstuck = antsstuck + 1; 
                actions = [];
                % Ant got stuck in dead-end
                if (been==0)
                    Been0 = Been0 + 1; 
                end
                if (been==1)
                    Maze(currentRow, currentColumn) = 0;
                    Been1 = Been1 + 1; 
                end
                if (been==4)
                    Maze(currentRow, currentColumn) = 0;
                    Been4 = Been4 + 1; 
                end
                if (been==3)
                    Maze(currentRow, currentColumn) = 0;
                    Been3 = Been3 + 1; 
                end
                if (been==2)
                    Been2 = Been1 + 2; 
                end
                break;
            end
            
            randomNumber = rand();
            
            % Go north
            if (randomNumber < (north / total))
                actions = [actions, 1];
                currentRow = currentRow - 1;
            % Go south 
            elseif (randomNumber < ((north + south) / total))
                actions = [actions, 3];
                currentRow = currentRow + 1;
            % Go west
            elseif (randomNumber < ((north + south + west) / total))
                actions = [actions, 2];
                currentColumn = currentColumn - 1;
            % Go east
            else
                actions = [actions, 0];
                currentColumn = currentColumn + 1;
            end
        end
        
        routeLength = size(actions,2);
        
        % The number of pheromones that will be added to all the paths this
        % ant has visited
        pherAddition = (PHEROMONES_DROPPED / routeLength);
        
        % In case ant made a loop, phero is cut to half.
        pherAddition = pherAddition * ownEnd;
        
        currentRow = startRow;
        currentColumn = startColumn;
        
        % Update the TempPerhNorth, south, east and west here based on the
        % actions 
        for k=1:routeLength
            action = actions(k);
            
            if (action == 0)
                % Went east
                TempPherEast(currentRow, currentColumn) = TempPherEast(currentRow, currentColumn) + pherAddition;
                currentColumn = currentColumn + 1;
            elseif (action == 1)
                % Went north
                TempPherNorth(currentRow, currentColumn) = TempPherNorth(currentRow, currentColumn) + pherAddition;
                currentRow = currentRow - 1;
            elseif (action == 2)
                % Went west
                TempPherWest(currentRow, currentColumn) = TempPherWest(currentRow, currentColumn) + pherAddition;
                currentColumn = currentColumn - 1;
            else
                % Went south
                TempPherSouth(currentRow, currentColumn) = TempPherSouth(currentRow, currentColumn) + pherAddition;
                currentRow = currentRow + 1;
            end
        end
        
        % If the routelength of this ant is shorter than the length of
        % shortestroute so far and is not 0 (which it is whene the ant got
        % stuck), this route is considered the new shortestRoute.
        if (routeLength < size(shortestRoute,2) && routeLength > 0)
            shortestRoute = actions;
        end
        
        % If the routelength is shorter than the convergence criterion,
        % break
        if (routeLength < CONVERGENCE_CRITERION && routeLength > 0)
            break;
        end;   
    end
    
    % stop when the shortestroute is shorter than a predefined criterion
    if (size(shortestRoute,2) < CONVERGENCE_CRITERION)
        break;
    end;
    
    % Evaporate some of the current pheromones. Afterwards, add the
    % pheromones from the previous iteration.
    PherNorth = (1-EVAPORATION_RATE) .* PherNorth;
    PherNorth = PherNorth + TempPherNorth;
    PherWest = (1-EVAPORATION_RATE) .* PherWest;
    PherWest = PherWest + TempPherWest;
    PherEast = (1-EVAPORATION_RATE) .* PherEast;
    PherEast = PherEast + TempPherEast;
    PherSouth = (1-EVAPORATION_RATE) .* PherSouth;
    PherSouth = PherSouth + TempPherSouth;
    
    % update the estimated path-length with the current shortest
    % PHEROMONES_DROPPED = size(shortestRoute,2);
    
    disp('Iteration: ');
    disp(i);
end

% write the results to file
dlmwrite(RESULT_FILE, []);
fileID = fopen(RESULT_FILE, 'wt'); 
results = [size(shortestRoute, 2), startLoc(1), startLoc(2)];
formatSpec = '%d;\n%d, %d;\n';
fprintf(fileID, formatSpec, results);
fprintf(fileID, '%d;', shortestRoute);
fclose(fileID);


disp('Ants that got stuck somehow:');
disp(antsstuck);
disp('Either: Ants that got stuck in death-end (been=1):');
disp(Been1);
disp('Or: Ants that got stuck winded in their own path (been=4):');
disp(Been4);
disp('Or: Stuck between wall element and own path (been=3):');
disp(Been3);
disp('Stuck with been=0 (shall be impossible):');
disp(Been0);
disp('Stuck with been=2: Hitting own tail more than once.');
disp(Been2);
disp('Done!');

disp(PherEast);
disp(PherNorth);
disp(PherWest);
disp(PherSouth);