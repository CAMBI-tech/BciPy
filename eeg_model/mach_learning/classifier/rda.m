classdef rda < handle
    % Regularized Discriminant Analysis
    %
    % It is called as the following,
    %
    % parameters.lambda=1;
    % parameters.gamma=0;
    % classifierObj = rda(parameters);
    % classifierObj.learn(data,labels);
    % scores=classifierObj.operate(data);
    
    %     properties (Constant)
    %         targetStrings={'target'};
    %     end
    
    properties (SetAccess = private)
        lambda = 1;
        gamma = 1;
        classes
        positiveLabel=1;
    end
    
    properties (SetAccess = private, GetAccess = private)
        means
        inverseCovariances
        logdeterminantOfCovariances
        priors
        positiveClassIndex
        covariances = [];
    end
    
    
    methods
        function self=rda(parameters)
            %parameters is the external parameter struct for the model. It may
            %contain parameters.lambda to set the shrinkage parameter (default = 1) and
            %parameters.gamma to set the regularization parameter (default = 1).
            %parameters.positiveLabel represent the label of the positive
            %class for the two class case (same type as the label vector elements). (default = 1)
            if(nargin > 0)
                if(isfield(parameters,'lambda'))
                    self.lambda=parameters.lambda;
                end
                if(isfield(parameters,'gamma'))
                    self.gamma=parameters.gamma;
                end
                if(isfield(parameters,'positiveLabel'))
                    self.positiveLabel=parameters.positiveLabel;
                end
            end
            
            
        end
        
        function learn(self,data,labels)
            % learn(data,labels,parameters) function learns the model, i.e. trains the classifier, from the data given
            %data is a d x N matrix, where d
            % is the number of dimensions and N is the number of samples. data
            % is used to learn the model corresponding to given parameters.
            %
            %labels is a 1 x N vector or 1 x N cell containing the labels
            %corresponding to samples.
            
            
            %             THIS IS GOING TO CONTAIN THE LEARNING CODES, i.e TRAINING OR CALIBRATION
            %
            [self.classes]=unique(labels);
            Nk=zeros(1,1,length(self.classes));
            
            
            d=size(data,1);
            
            self.inverseCovariances=cell(1,length(self.classes));
            self.logdeterminantOfCovariances=zeros(1,length(self.classes));
            
            Sk=cell(1,length(self.classes));
            S=zeros(d,d);
            
            self.means=cell(1,length(self.classes));
            self.positiveClassIndex=length(self.classes);
            
            for(classIndex=1:length(self.classes))
                if(isnumeric(labels))
                    classLocations=(labels==self.classes(classIndex));
                    if(self.classes(classIndex)==self.positiveLabel)
                        self.positiveClassIndex=classIndex;
                    end
                elseif(iscell(labels))
                    classLocations=(strcmpi(labels,self.classes(classIndex)));
                    if(strcmpi(self.classes(classIndex),self.positiveLabel))
                        self.positiveClassIndex=classIndex;
                    end
                end
                
                
                classData=data(:,classLocations);
                Nk(classIndex)=size(classData,2);
                self.means{classIndex}=mean(classData,2);
                meanSubtractedClassData=bsxfun(@minus,classData,self.means{classIndex});
                Sk{classIndex}=meanSubtractedClassData*meanSubtractedClassData.';
                S=S+Sk{classIndex};
            end
            
            N=sum(Nk);
            self.priors=Nk/N;
            Nk=((1-self.lambda)*Nk+self.lambda*N);
            %S=S*self.lambda;
            
            %Sk=bsxfun(@plus,Sk,S);
            
            %invNk=1./((1-self.lambda)*Nk+self.lambda*N);
            %Ck=bsxfun(@times,Sk,invNk);
            
            %cellfun(Ck)
            
            for(classIndex=1:length(self.classes))
                
                shrinkedCov=((1-self.lambda)*Sk{classIndex}+S*self.lambda)/Nk(classIndex);
                Sk{classIndex}=[];
                
                [Q,R]=qr((1-self.gamma)*shrinkedCov + (self.gamma/d*trace(shrinkedCov))*eye(d));
                
                self.inverseCovariances{classIndex}=R\(Q.');
                self.logdeterminantOfCovariances(classIndex)=sum(log(abs(diag(R))));
                
            end
            
            
            
        end
        
        function output=operate(self,data)
            % operate(data) function tests or projects the new samples given by
            % data. It requires that the learn function had run.
            %
            %INPUT data is a d x N matrix, where d
            % is the number of dimensions and N is the number of samples. data
            % is used to learn the model corresponding to given parameters.
            %
            %INPUT output is an 1 x N vector containing the scores obtained after testing.
            
            %THIS IS GOING TO CONTAIN THE OPERATION, i.e TESTING, PROJECTION
            
            Nc=length(self.classes);
            N=size(data,2);
            tempOutput=zeros(Nc,N);
            for classIndex=1:Nc
                meanSubtractedData=bsxfun(@minus,data,self.means{classIndex});
                tempOutput(classIndex,:)=-(sum((meanSubtractedData.*(self.inverseCovariances{classIndex}*meanSubtractedData)),1)+self.logdeterminantOfCovariances(classIndex))/2+log(self.priors(classIndex));
            end
           % if(Nc==2)
            %    outputMultiplier=[-1,1]*((-1)^self.positiveClassIndex);
            %    output=outputMultiplier*tempOutput;
            %else
                output=tempOutput;
            %end
        end
        
        function means = getMeans(self)
            means = self.means;
        end
        
        function invCovs = getInvCovariances(self)
            invCovs = self.inverseCovariances;
        end
        
        function covs = getCovariances(self)
            if isempty(self.covariances)
                for ii = 1:numel(self.inverseCovariances)
                    self.covariances{ii} = inv(self.inverseCovariances{ii});
                end
            end
            covs = self.covariances;
        end
    end
    
    
    
end
