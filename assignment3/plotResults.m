clc;
clear all;

easyMazePheromones = [20, 40, 60, 80, 100, 120, 140];
easyMazeResults = [84, 60, 56, 60, 66, 78, 126];

mediumMazePheromones = [100, 200, 300, 400, 500, 600, 700];
mediumMazeResults = [269, 237, 235, 197, 217, 232, 287];

hardMazePheromones = [100, 300, 500, 700, 900, 1100, 1300];
hardMazeResults = [1061, 1031, 1019, 1143, 1261, 1005, 937];

insaneMazePheromones = [100, 300, 500, 700, 900, 1100, 1300];
insaneMazeResults = [0, 0, 805, 877, 851, 0, 0];

%plot(easyMazePheromones, easyMazeResults)
%plot(mediumMazePheromones, mediumMazeResults)
%plot(hardMazePheromones, hardMazeResults)
%plot(insaneMazePheromones, insaneMazeResults)