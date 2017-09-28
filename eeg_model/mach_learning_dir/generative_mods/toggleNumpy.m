function outArray = toggleNumpy(inArray, varargin)
% matlab is a bit lame when it comes to converting to numpy, it only takes
% vectors:
%
% http://www.mathworks.com/help/matlab/matlab_external/passing-data-to-python.html
%
% this function toggles and array between a numpy and MATLAB state

p = inputParser;
p.addParameter('verboseFlag', true, @islogical);
p.parse(varargin{:});

if isnumeric(inArray)
    % MATLAB input given, convert to row major by flipping permute vector
    origSize = size(inArray);
    dimVector = 1:length(origSize);
    
    % Flip columns and rows because of ordering scheme
    [origSize(1), origSize(2)] = deal(origSize(2), origSize(1));
    [dimVector(1),dimVector(2)] = deal(dimVector(2),dimVector(1));
    
    % Permute to change array layout
    inArray = permute(inArray,dimVector);
    outArray = py.numpy.array(inArray(:).');
        
    % Flip original shape because numpy's reshape reads inputs in opposite
    % order
    outArray = outArray.reshape(fliplr(origSize));
    return
end

% Python numpy array given, convert to MATLAB array

dim = double(py.len(inArray.shape));

% http://www.mathworks.com/matlabcentral/answers/157347-convert-python-numpy-array-to-double
% d is for double, see link below on types
outArray = double(py.array.array('d', py.numpy.nditer(inArray)));
shape = double(py.array.array('d', py.list(inArray.shape)));

switch dim
    case 1
        % python has 1d arrays (all of MATLABs arrays are at least 2d),
        % for this reason there is ambiguity as to whether we should
        % make a row or column vector out of a 1d array ...
        if p.Results.verboseFlag
            warning('1d numpy array passed, building col vector (maybe input was row?)');
        end
        outArray = outArray(:);
    otherwise        
        % convert numpy array to column major compatible with matlab
        dimVector = 1:dim;

        % Flip columns and rows because of ordering scheme
        [dimVector(1),dimVector(2)] = deal(dimVector(2),dimVector(1));
        outArray = permute(reshape(outArray, fliplr(shape)), dimVector);    
end

end