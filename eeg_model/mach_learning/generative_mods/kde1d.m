classdef kde1d < handle
%%  A kde1d is an object that contains a data vector that we want to calculate its kernel density distribution. When a
%   kde1d is constructed, kernel width will be calculated using the input data. The constructor returns a handle to this object.
%
%  kde1d properties:
%   kernelWidth         - sigma of normal distribution fitted on each data
%                         point for kernel density estimation.(formula?)
%   data                - a vector that we want to calculate its kernel
%                         density ditribution.(In this project is usually vector of scores.)
% 
%  kde1d methods:
%   probs        - Estimates kernel density distribution of a given data


   properties
   kernelWidth 
   data
   end
 %% Methods of the kde1d class
   methods
        %% kde1d(data)
        % Constructor for the kde1d object. It initializes the empty 
        % kdeld object with data and the kernelWidth parameter
        function self=kde1d(data)
            self.kernelWidth=1.06*min(std(data),iqr(data)/1.34)*length(data)^-0.2; % formula?
            if(size(data,1)==1)
                self.data=data';
            else
                self.data=data;
            end
        end
        %% probs(x)
        % probs(x)function estimates overall kernel for a given x vector.It fits a normal kernel on each point of data 
%        (mean of this normal distribution is on each data point and sigma is constant and equal to kernel width)
%        and total distribution function is equal to sum of all these fitted
%        kernels on each data point.    
        function p=probs(self,x)
            p=zeros(length(x),1);
            for(xi=1:length(x))
                p(xi)=sum(normpdf(x(xi),self.data,self.kernelWidth));
            end
            p=p/length(self.data);
            
        end
        
        function x=getSample(self,sampleCount)
            x=randn(sampleCount,1)*self.kernelWidth+self.data(randi(length(self.data),[sampleCount,1]));
        end
    end
    
    
    
end