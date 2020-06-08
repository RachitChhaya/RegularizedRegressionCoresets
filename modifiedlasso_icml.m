%This is the code we used for our modified lasso experiments. We have
%provided comments to make it readable

%% Data Generation 
% We actually generated the matrices using the code below and saved the mat files for later use
clear;
clc;
n=100000;  % no. of rows
d=30;      % no of columns

%NG Matrix created as suggested in Yang et al. 2015
alpha1=0.00065;
NG=[alpha1*randn(n-d/2,d/2) (10^-8)*rand(n-d/2,d/2);zeros(d/2,d/2) eye(d/2)];

xoriginal=sprand(d,1,0.3);% sparse vector for generating data. Works with any other vector too
xoriginal=full(xoriginal);   

A=NG;  
b=A*xoriginal;
err=randn(n,1);
b=b + (10^-5)*norm(b)/norm(err)*err;

lambda=0.5;   % lambda (regularization parameter). You can change the value here
%% patternsearch implementation to solve modified lasso

x0=rand(d,1);  %initial value of solution vector for pattern search matlab function

%options for pattern search
options=optimoptions('patternsearch','MaxFunctionEvaluations',1000000,'MaxIterations',25000,'UseParallel',true,'Display','iter');

%% To do the sparsity experiment run the below code in a loop for various 
%lambda values and note sparsity of each 
%solution vector for each value of lambda. For other experiment comparing
%coreset for different sample sizes comment the lasso  and ridge solutions.
%and run for each lambda separately. You have to check the coordinate values of vector and conver them to zero if absolute value less than 10^-6. 
%For more details please see the experiment section in the paper.

%modified lasso function definition
modifiedlasso = @(x)norm(A*x-b)^2 + lambda*(norm(x,1)^2);  %modified lasso

% Lasso definition for sparsity comparison experiment 
Lasso = @(x)norm(A*x-b)^2 + lambda*(norm(x,1));



%solving modified lasso using pattern search
[xmodlasso,fval]=patternsearch(modifiedlasso,x0,[],[],[],[],[],[],[],options);
%solving lasso using pattern search
[xlasso,fvallasso]=patternsearch(Lasso,x0,[],[],[],[],[],[],[],options);
%ridge solution
xridge=((A'*A+lambda*eye(d))^(-1))*A'*b;


%% Sampling for modified lasso  five experiments
relative_errorvector=zeros(5,1);  % vector to store relative error for each experiment
funcvaluesampled=zeros(5,1); % vector to store function value for each experiment


probnum=zeros(n,1);  %sensitivity score vetor

%Ridge leverage scores

Aappend=[A;sqrt(lambda)*eye(d)];
[U,S,V]=svd(Aappend,'econ');
U1=U(1:n,:);

for l=1:5
    samplesize=200;
%storing sketched loss  reweighing with inverse of prob and samplesize                              %storing values of sketched loss relative
    sampledA=zeros(samplesize,d);  % coreset matrix intialization
    sampledb=zeros(samplesize,1);

%% Two Sampling strategies. Use one at a time.Comment the other for loop of the other.

    %%Sampling using ridge leverage scores
     for k=1:n
            probnum(k)= norm(U1(k,:),2)^2;      %vector of ridge leverage scores
     end

%     %% Sampling using Uniform sampling
%     for k=1:n
%          probnum(k)= (1);
%     end


%% Sampling is done here

    probvec =probnum/norm(probnum,1);% probability vector
    [val,in] = sort(probvec,'ascend');%sort probabilities in ascending order
    sample = cumsum(val);  % cumulative sum of probabilities
    
    for k=1:samplesize
        index=find(sample > rand(),1);
        sampledA(k,:)=A(in(index),:)*(1/sqrt(samplesize*probvec(in(index))));
        sampledb(k,:)=b(in(index))*(1/sqrt(samplesize*probvec(in(index))));          
    end

%% Solving modified lasso for sampled data
    samplemodifiedlasso= @(x)norm(sampledA*x-sampledb)^2 + lambda*(norm(x,1)^2);
    [xsampledmodifiedlasso,fvalsamplemodifiedlasso]=patternsearch(samplemodifiedlasso,x0,[],[],[],[],[],[],[],options);

    %plugging vector obtained from smaller data with full data
    fval2=modifiedlasso(xsampledmodifiedlasso);   
    
    %relative error as defined in the paper
    error_relative=abs(modifiedlasso(xmodlasso)-modifiedlasso(xsampledmodifiedlasso))/modifiedlasso(xmodlasso);
    
    relative_errorvector(l,1)=error_relative;  %storing values of relative error
    funcvaluesampled(l,1)=fval2;
end
Median=median(relative_errorvector); % Median of 5 experiments reported in the paper

%% 
