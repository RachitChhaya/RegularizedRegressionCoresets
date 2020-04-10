%This is the code we used for our RLAD (Regularized Least Absolute Deviation) experiments. We have
%provided comments to make it readable

%% Data Generation  %%Same as for modified lasso as we used the same NG matrix
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
Acentered=A-repmat(mean(A),size(A,1),1); %centering A for taking care of bias term
Afinal= Acentered ./ repmat(var(A),size(A,1),1);
A= Afinal; %data matrix

b=A*xoriginal;
err=randn(n,1);
b=b + (10^-5)*norm(b)/norm(err)*err;

lambda=0.5;   % lambda (regularization parameter). You can change the value here
%% patternsearch implementation to solve modified lasso

x0= pinv(A'*A)*A'*b; %initial value of solution vector for pattern search matlab function. Gives faster convergence empirically

%options for pattern search
options=optimoptions('patternsearch','MaxFunctionEvaluations',1000000,'MaxIterations',25000,'UseParallel',true,'Display','iter');


%% Solving RLAD using patternsearch
%RLAD function defintion
l1lossl1regfn = @(x)norm(A*x-b,1) + lambda*norm(x,1);  %l1lossl1reg 

[xsoln,fval]=patternsearch(l1lossl1regfn,x0,[],[],[],[],[],[],[],options);



%% %% Sampling for RLAD  five experiments
relative_errorvector=zeros(5,1);   % vector to store relative error for each experiment
funcvaluesampled=zeros(5,1);  % vector to store function value for each experiment

%% Well conditioned basis for l1 norm
S= randn(1000,n)./randn(1000,n);
SA=S*A;
[Q,R]=qr(SA,0);
U=A*pinv(R);  %well conditioned basis of A
normA=norm(A,1);  %Induced 1 norm of A

for l=1:5
    samplesize=200;
%storing sketched loss  reweighing with inverse of prob and samplesize                              %storing values of sketched loss relative
    sampledA=zeros(samplesize,d);  % coreset matrix initialization
    sampledb=zeros(samplesize,1);

    probnum=zeros(n,1); % vector of sensitivities
      
%% Two  Sampling strategies. Use one at a time. Comment the other for loop of the other.

%% Sampling using our sensitivity scores

    for k=1:n
        probnum(k)=((norm(U(k,:),1))/(1+(lambda/normA)))+(1/n);     %vector of  sensitivity scores
    end


%% Sampling using Uniform sampling
%     for k=1:n
%         probnum(k)= (1);
%     end


%% Actual Sampling occurs here
    probvec =probnum/norm(probnum,1);% probability vector
    [val,in] = sort(probvec,'ascend');%sort probabilities in ascending order
    sample = cumsum(val);  % cumulative sum of probabilities
    for k=1:samplesize
        index=find(sample > rand(),1);
        sampledA(k,:)=A(in(index),:)*(1/(samplesize*probvec(in(index))));
        sampledb(k,:)=b(in(index))*(1/(samplesize*probvec(in(index))));          
    end
%% Solving RLAD for sampled data
    samplel1lossl1reg= @(x)norm(sampledA*x-sampledb,1) + lambda*(norm(x,1));
    [xsampledsoln,fvalsampled]=patternsearch(samplel1lossl1reg,x0,[],[],[],[],[],[],[],options);
    
    %plugging vector obtained from smaller data with full data
    fval2=l1lossl1regfn(xsampledsoln);
    %plugging vector obtained from smaller data with full data
    error_relative=abs(l1lossl1regfn(xsampledsoln)-l1lossl1regfn(xsoln))/l1lossl1regfn(xsoln);
    
    relative_errorvector(l,1)=error_relative; %storing values of relative error for each experiment
    funcvaluesampled(l,1)=fval2;
end
Median=median(relative_errorvector); % Median of 5 experiments reported in the paper
%% 