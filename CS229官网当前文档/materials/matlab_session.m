%%%%%%%%%%%%%%%%%%%%%%%
%% CS229, Autumn 2015-16
%% Matlab Review Session
%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all
% clear console, clear all variables, close all plot windows

%%%%%%%%%%%%%%%%%%%%%%
%% Basic operations %%
5+6
3-2
5*8
1/2
2^6
1 == 2  % false (0)
1 ~= 2  % true (1).  note, not "!="
1 && 0  % Any non-zero number is considered as "true"
1 || 0
xor(1,0)
i, j, 1i, 1j % complex imaginary unit  - i and j can be overwritten as variables
pi % pi number with high precision

%% variable assignment
a = 3 % semicolon suppresses output
b = 'hello', b(1) % Store character string
c = 3>=1 % Store the result of comparison (true or 1)

%% Displaying them:
disp(sprintf('2 decimals: %0.2f', pi)) % display number with 2 decimals
fprintf('6 decimals: %0.6f\n', pi) % display number with 6 decimals
format long % Long format with many decimals
pi
format short % Short formal with less decimals
pi

who
whos

clear
who

%% Documentation
help xor
doc xor

clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Vectors and matrices %%
v = [1 2 3]
v'             % conjugate transpose
v.'            % transpose
v = 1:0.1:2  % from 1 to 2, with stepsize of 0.1. Useful for plot axes
v = 1:6        % from 1 to 6, assumes stepsize of 1
v = linspace(1,6,6)
A = [1 2; 3 4; 5 6]

w = 2*ones(2,3)  % same as C = [2 2 2; 2 2 2]
w = ones(1,3)    % 1x3 vector of ones
w = zeros(1,3)
w = rand(1,3)    % 1x3 vector drawn from a uniform distribution on [0,1] 
w = randn(1,3)   % 1x3 vector drawn from a normal distribution (mean=0, var=1)
w = -6 + sqrt(10)*(randn(1,10000));  % (mean = 1, var = 2)
hist(w)
e = [];       % empty vector
isempty(e)
I = eye(4)    % 4x4 identity matrix
I = diag(ones(4,1))
diag(I)

clc, close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vectors and matrices- dimensions %%
sz = size(A)
size(A,1)          % number of rows
size(A,2)          % number of cols
length(A)          % size of the longest dimension
numel(A)	       % number of elements

%% submatrix access
A(3, 2)         % (row, column), 1-based
A(2, :)			% get second row
A(:, 2)			% get second column
A(1, end)       % first row, last element
A(end, :)		% last row
A(2:end,:)		% get all but first row
A([1 3],:)      % first and third rows
A(:)			% returns all the elements of A as column

A(:,2) = [10 11 12]'  % change second column
[A, [100; 101; 102]] % append column vec
[ones(size(A,1),1),  A]  % e.g bias term in linear regression

clc
%%%%%%%%%%%%%%%%%%%%%%%
%% Matrix operations %%
B = [0 1; 1 0; 1 1] % same dims as A
C = [5 6; 7 8] % same dims as A
A * C % matrix multiplication
A .* B % element-wise multiplcation
% A .* C  or A * B gives error - wrong dimensions
A .^ 2 % elementwise power
C ^ 2 % matrix power
1 ./ A % elementwise division

A / B  % multiplication by pseudo-inverse of B, matrices must be compatible
A \ B  % multiplication by pseudo-inverse of A, matrices must be compatible
A & B  % different from A&&B, A and B can be matrices of same dimensions
A | B  % different from A||B, A and B can be matrices of same dimensions

v + ones(1,length(v))
v + 1  % same

clc
%%%%%%%%%%%%%%%%%
%% Cell arrays %%
C = cell(3) % n * n square cell
sz1 = 2; sz2 = 3; sz3 = 5;
cell(sz1, sz2, sz3) % cell of size sz1 * sz2 * sz3

C{1, 2} = [1]

C{1:2}
C(1:2)

clc
%%%%%%%%%%%%%%%%%%%%%%
%% Useful functions %%
log(v)        		% natural logarithm, element-wise operation
exp(v)        		% exponential
abs(v)                  % absolute values
max(v), min(v)     	% returns [value, index]
[val,ind] = max(v)
D = [2 3 4];
find(D > 2)        	% Find all non-zero elements
sum(A, 1) 		% sum columns (default)
sum(A, 2)		% sum rows
sum(A(:))               % sum all elements of A
sum(sum(A))             % same
inv(A(1:2,1:2))		% inverse
pinv(A)                 % pseudoinverse, inv(A’*A)*A’
reshape(A, [2 3])
tic % starts stopwatch
toc % stop stopwatch and returns time elapsed since last "tic"
% WARNING: don’t overwrite function names.

clc
%%%%%%%%%%%%%%%%%%
%% Control Flow %%
summation = 0;
for i = 1 : 100
   i
   summation = summation + i;
   if (i == 99) 
       break;
   elseif(i == 98)
       continue;
   else
       continue;
   end
end
summation
sum(1 : 99)
% Same as sum(1 : 99)

A = 1 : 100;
i = 1;
summation = 0;
while (i <= numel(A))
    summation = summation + A(i);
    i = i + 1;
end
summation
sum(A)
% Same as sum(1 : 100)

% Prefer Matrix operation over for-loop
% www.quora.com/What-are-good-ways-to-write-matlab-code-in-matrix-way

clc, clear all
%%%%%%%%%%%%%%%%%%
%% loading data %%
load q1y.dat
load q1x.dat
who
whos

[th,ll] = logistic_grad_ascent(q1x,q1y);
plot(ll)

clear q1y  % clear w/ no argt clears all
v = q1x(1:10);
save hello v;   % save variable v into file hello.mat
save hello.txt v -ascii; % save as ascii
% fopen, fprintf, fscanf also work
% ls  %% cd, pwd  & other unix commands work in matlab; to access shell,
% preface with "!" 

%%%%%%%%%%%%%%
%% Plotting %%
close all; clear all; clc;
A = 1 : 100;
B = rand(1, 100);
C = rand(1, 100);
figure(); % Open figure window
plot(A, B, 'b-o','linewidth', 1.5); % Plot B(A) with drawing options
hold on; % This prevents the second plot() call from erasing the first curve
plot(A, C, 'm-*','linewidth', 1.5); % Plot C(A) with drawing options
xlabel('myXAxis'); ylabel('myYAxis'); % X- and Y-axis labels
legend('myA', 'myB'); % legend names
title('myPlot'); % Title name for figure (on top)
saveas(gcf, 'myPlot', 'epsc'); % Save figure as .eps

%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting - subplot %%
close all; clear all; clc;
A = 1 : 100;
B = rand(1, 100);
C = rand(1, 100);
figure(); % Open figure window
subplot(2, 1, 1); % In 2x1 subplots, draw in 1st plot
plot(A, B, 'b-o','linewidth', 1.5); % Plot B(A) with drawing options
xlabel('myXAxis'); ylabel('myYAxis'); % X- and Y-axis labels
legend('myA'); % legend
subplot(2, 1, 2);% In 2x1 subplots, draw in 2nd plot
plot(A, C, 'm-*','linewidth', 1.5); % Plot C(A) with drawing options
xlabel('myXAxis'); ylabel('myYAxis'); % X- and Y-axis labels
legend('myB'); % legend names
saveas(gcf, 'myPlot', 'epsc'); % Save figure as .eps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Other plotting functions %%
% plot() - plot 2-D line data
% plot3() - plot 3-D line data			
% scatter() - plot 2-D data as scattered data points			
% scatter3() - plot 3-D data as scattered data points
% loglog() - plot 2-D line data with log-scaled X and Y axis
% semilogx() - plot 2-D line data with log-scaled X axis
% semilogy() - plot 2-D line data with log-scaled Y axis
% histogram() - aggregate data as histogram
doc plot % More info on plot options

close all; clc;
%%%%%%%%%%%%%%
%% Data input and output
save('myWorkspace')     % save the whole workspace
save('myA', 'A') 		% save the specified variable
load('myWorkspace')	
load('myA')

% csvread()  	% read a comma-separated value file into a matrix
% dlmread()  	% read an ASCII-delimited numeric data file into a matrix
% textscan() 	% manual input processing	

% csvwrite()	  % write numeric data in a matrix into file as comma-separated values
% dlmwrite() 	  % write numeric data in a matrix to an ASCII format file
% fprintf()	  % manual output processing
% saveas(gcf, 'myPlot', 'epsc')

% fprintf()  
fprintf('I scored %d in %s!\n', 100, 'CS 229')
disp(sprintf('I scored %d in %s!\n', 100, 'CS 229'))

clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reshape and replication %%
A = magic(3)
A = [A [0;1;2]] % Adds a column [0;1;2]
reshape(A,[4 3]) % Reshape A as a 4x3 matrix
A = reshape(A,[2 6]) % Reshape A as a 2x6 matrix
v = [100;0]
bsxfun(@plus, A, v) % Adds v to each column of A
help bsxfun
