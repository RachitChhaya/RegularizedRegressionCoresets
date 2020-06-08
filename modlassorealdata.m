%This is the code we used for our modified lasso experiments with realdata. We have
%provided comments to make it readable

clear;
clc;
%% Reading Realdata set and Initializing matrix
T =readtable('Folds5x2_pp.xlsx','Range','A2:E9569');

%T = readtable('CASP.csv', 'HeaderLines',1);

T= table2array(T);
%b=T(:,1); For CASP

b=T(:,5);
%A=T(:,2:10); For CASP

A=T(:,1:4);
[n,d]=size(A);

%%Normalizing the data

%A=normalize(A,'norm',Inf);
for i=1:d
    maxval=max(A(:,i));
    for j=1:n
        A(j,i)=A(j,i)/maxval;
    end
end


lambda=1;   % lambda (regularization parameter). You can change the value here
% %% patternsearch implementation to solve modified lasso
% 
%x0=pinv(A'*A)*A'*b;%initial value of solution vector for pattern search matlab function
% 
x0=rand(d,1);
%options for pattern search
options=optimoptions('patternsearch','MaxFunctionEvaluations',1000000,'MaxIterations',25000,'UseParallel',true,'Display','iter');

%% To do the sparsity experiment run the below code in a loop for various 
%lambda values and note sparsity of each 
%solution vector for each value of lambda. For other experiment comparing
%coreset for different sample sizes comment the lasso  and ridge solutions.
%and run for each lambda separately

%modified lasso function definition
modifiedlasso = @(x)(1/(2*n))*(norm(A*x-b)^2 + lambda*(norm(x,1)^2));  %modified lasso


%solving modified lasso using pattern search
[xmodlasso,fval]=patternsearch(modifiedlasso,x0,[],[],[],[],[],[],[],options);

%%

%% Sampling for modified lasso  five experiments
relative_errorvector=zeros(5,1);  % vector to store relative error for each experiment
funcvaluesampled=zeros(5,1); % vector to store function value for each experiment
vectdiffvector=zeros(5,1);

probnum=zeros(n,1);  %sensitivity score vetor

%Ridge leverage scores

Aappend=[A;sqrt(lambda)*eye(d)];
[U,S,V]=svd(Aappend,'econ');
U1=U(1:n,:);

for l=1:5
    samplesize=300;
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
    samplemodifiedlasso= @(x)(1/(2*samplesize))*(norm(sampledA*x-sampledb)^2 + lambda*(norm(x,1)^2));
    [xsampledmodifiedlasso,fvalsamplemodifiedlasso]=patternsearch(samplemodifiedlasso,x0,[],[],[],[],[],[],[],options);

    %plugging vector obtained from smaller data with full data
    fval2=modifiedlasso(xsampledmodifiedlasso);   
    
    %relative error as defined in the paper
    error_relative=abs(modifiedlasso(xmodlasso)-modifiedlasso(xsampledmodifiedlasso))/modifiedlasso(xmodlasso);
    
    %difference of two vectors
    vectorrelative=norm(xmodlasso-xsampledmodifiedlasso)/norm(xmodlasso);
    vectdiffvector(l,1)=vectorrelative;
    relative_errorvector(l,1)=error_relative;  %storing values of relative error
    funcvaluesampled(l,1)=fval2;
end
Median=median(relative_errorvector); % Median of 5 experiments reported in the paper
Median2=median(vectdiffvector);
%% 
%%

    

