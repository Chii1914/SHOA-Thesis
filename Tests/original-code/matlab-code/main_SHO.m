%_______________________________________________________________________________________%
%  Sea-Horse optimizer (SHO)
%  Developed in MATLAB R2018a
%
%  programmer: Shijie Zhao and Tianran Zhang   
%  E-mail: zhaoshijie@lntu.edu.cn
%          ztr20010118@126.com
%  The code is based on the following papers.
%  Shijie Zhao, Tianran Zhang, Shilin Ma, Mengchen Wang 
%  Sea-horse optimizer: a novel nature-inspired meta-heuristic for global
%  optimization problems.
%  Applied Intelligence
%_______________________________________________________________________________________%
clear all 
clc

popsize=30; % Number of search agents
Max_iter=500; % Maximum iteration
F_name='F1'; % Name of the test function that can be from F1 to F23 (Table 2,3,4 in the paper)

[LB,UB,Dim,fobj]=BenchmarkFunctions(F_name);% Load details of the selected benchmark function

tic
[ObjectiveFitness,ObjectivePosition,Convergence_curve,Trajectories,fitness_history, population_history]=SHO(popsize,Max_iter,LB,UB,Dim,fobj);
time=toc;

semilogy(1:Max_iter,Convergence_curve,'color','r','linewidth',2.5);
title('Convergence curve');
xlabel('Iteration');
ylabel('Best score obtained so far')
display(['The running time is:', num2str(time)]);
display(['The best solution obtained by SHO is : ', num2str(ObjectiveFitness)]);
display(['The best optimal sea horse of the objective funciton found by SHO is : ', num2str(ObjectivePosition)]);