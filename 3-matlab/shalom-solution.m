clear all
clc
%close all

N = 10000;
n = 100;
m = sqrt(n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%   Initialization of Matrix D
D = zeros(2*m,n);
for qq = 1:m;
    D(qq,m*(qq-1)+1:m*(qq-1)+m) = ones(1,m);
end
for qq = 1:m;
    D(m+1:2*m,m*(qq-1)+1:m*(qq-1)+m) = eye(m);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%=Experiments
for kk = 1:N;
    Picks = randsample(n,n);
    Sticks = zeros(n,1);
    for ii = 1:n
        Sticks(Picks(ii)) = 1;
        if sum(D*Sticks==m)>=1
            break;
        end
    end
    NumberOfSticks(kk) = ii; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%         Analysis
mean(NumberOfSticks)
