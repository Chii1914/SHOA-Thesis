Sea-Horse optimizer (SHO)
 
August 6, 2022

email: zhaoshijie@lntu.edu.cn 
       ztr20010118@126.com

The files in this zip archive are MATLAB m-files that can be used to study Sea-Horse optimizer algorithm.

SHO is the method that we invented and wrote about in the following paper:
Shijie Zhao, Tianran Zhang, Shilin Ma, and Mengchen Wang, Sea-Horse optimizer.

The MATLAB files and their descriptions are as follows:
BenchmarkFunctions.m
This is the benchmark functions discussed in the paper. You can use it as template to write your own function if you are interested in testing or optimizing some other functions. 
SHO.m
This is the core code of the SHO algorithm.
initialization.m
This contains various initialization settings for the optimization methods. You can edit this file to change the population size, the generation count limit, the problem dimension, the maximum Function Evaluations (FEs), and the percentage of population of any of the optimization methods that you want to run.
levy.m
This is used to generate the Levyflight random coefficient. 
function_plot.m 
This function draw the search space for benchmark functions.
Function_contour.m
This function draw the contour for benchmark functions.
I hope that this software is as interesting and useful to you as is to me. Feel free to contact me with any comments or questions.



