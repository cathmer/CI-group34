function route = antFunction(Maze, startColumn, startRow, endColumn, endRow)

% Maximum # of iterations before the algorithm terminates
MAX_ITERATIONS = 500;
% # of ants per iteration
ANTS_PER_ITERATION = 5;
% This is a guess of the length of the route and is a multiplier for the
% amounts of pheromomens that are dropped
PHEROMONES_DROPPED = 1000;
% The rate at which pheromones evaporate
EVAPORATION_RATE = 0.1;
% The route length for which the algorithm is satisfied
CONVERGENCE_CRITERION = 200;
% The alpha and beta of the probability function. Control the relative
% influences of pheromones and path length
ALPHA = 1;
BETA = 0.5;

% Vector which will store the actions of the shortest route
shortestRoute = zeros(1, size(Maze,1) * size(Maze,2));

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
            % Set the current location to 0, to indicate the ant has visited it
            TempMaze(currentRow, currentColumn) = 0;
            
            options = 0;
            
            % Determine the probabilities for each adjacent square
            if (currentRow == 1)
                north = 0;
            else
                north = PherNorth(currentRow, currentColumn)^ALPHA * TempMaze(currentRow - 1, currentColumn)^BETA;
            end
            
            if (currentRow == size(Maze,1))
                south = 0;
            else
                south = PherSouth(currentRow, currentColumn)^ALPHA * TempMaze(currentRow + 1, currentColumn)^BETA;
            end
            
            if (currentColumn == 1)
                west = 0;
            else
                west = PherWest(currentRow, currentColumn)^ALPHA * TempMaze(currentRow, currentColumn -1)^BETA;
            end
            
            if (currentColumn == size(Maze,2))
                east = 0;
            else
                east = PherEast(currentRow, currentColumn)^ALPHA * TempMaze(currentRow, currentColumn + 1)^BETA;
            end
            
            total = north + south + west + east;
            
            % No path, so break and empty actions
            if total == 0
                actions = [];
                Maze(currentRow, currentColumn) = 0;
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
    
    PherNorth = (1-EVAPORATION_RATE) .* PherNorth;
    PherNorth = PherNorth + TempPherNorth;
    PherWest = (1-EVAPORATION_RATE) .* PherWest;
    PherWest = PherWest + TempPherWest;
    PherEast = (1-EVAPORATION_RATE) .* PherEast;
    PherEast = PherEast + TempPherEast;
    PherSouth = (1-EVAPORATION_RATE) .* PherSouth;
    PherSouth = PherSouth + TempPherSouth;
end

route = shortestRoute;

end