clear all
%close all
clear classes

% Compares the results for a FIR linear phase bandpass filter between
% MATLAB implementation and the Python implementation. Generates  
% screens freq. response in both phase and amplitude

mod1 = py.importlib.import_module('filters');
py.reload(mod1);

fs = 256;

Fstop1 = 0.07;             % First Stopband Frequency
Fpass1 = 2.15;             % First Passband Frequency
Fpass2 = 40;               % Second Passband Frequency
Fstop2 = 44;               % Second Stopband Frequency
Dstop1 = 0.0039810717055;  % First Stopband Attenuation
Dpass  = 0.19879359342;    % Passband Ripple
Dstop2 = 0.0039810717055;  % Second Stopband Attenuation
dens   = 20;               % Density Factor

% Calculate the order from the parameters using FIRPMORD.
bands = [Fstop1 Fpass1 Fpass2 Fstop2]/(fs/2);
gain = [0,1,0];
[n,fo,ao,w] = 	firpmord(bands, gain, [Dstop1 Dpass Dstop2]);
b = firpm(n,fo,ao,w,{dens});

bands = [0, bands, 0.5 * 2]/2;
% Unfortunately the python code can cross validate and update filter length
% based on specification. Assuming filter is going to be optimal and stays
% the same we can accept the length is known.
r_val = py.filters.test_filter(toggleNumpy(n),toggleNumpy(bands),toggleNumpy(gain),toggleNumpy(dens));
b_py = toggleNumpy(r_val);


% Plot frequency amplitude and phase responses of the filters
% We can observe that freq. amp. response of the MATLAB filter is a bit
% better. This is because signal processing tollbox has the strong hand
% over python.
figure()
freqz(b,1,1024,fs)
title('MATLAB implementation')
figure()
freqz(b_py,1,1024,fs)
title('Python implementation')

% Plot group delays of each implementation
% We can obserce that the group delaty of the filters are both stay the
% same throughout the freuency domain. The signal does not detoriate.
figure()
subplot(2,1,1)
grpdelay(b,1)
title('MATLAB implementation')
subplot(2,1,2)
grpdelay(b_py,1)
title('Python implementation')