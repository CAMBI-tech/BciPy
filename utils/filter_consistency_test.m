clear all
close all
clear classes

% Compares the results for a FIR linear phase bandpass filter between
% MATLAB implementation and the Python implementation. Generates  
% screens freq. response in both phase and amplitude

mod1 = py.importlib.import_module('filters');
py.reload(mod1);

fs = 300;

Fstop1 = 0.07;             % First Stopband Frequency
Fpass1 = 2.15;             % First Passband Frequency
Fpass2 = 40;               % Second Passband Frequency
Fstop2 = 44;               % Second Stopband Frequency
Dstop1 = 0.0039810717055;  % First Stopband Attenuation
Dpass  = 0.19879359342;    % Passband Ripple
Dstop2 = 0.0039810717055;  % Second Stopband Attenuation
dens   = 10;               % Density Factor

% Calculate the order from the parameters using FIRPMORD.
bands = [Fstop1 Fpass1 Fpass2 Fstop2]/(fs/2);
gain = [0,1,0];
[n,fo,ao,w] = 	firpmord(bands, gain, [Dstop1 Dpass Dstop2]);
b = firpm(n,fo,ao,w,{dens});

bands = [0, bands, 0.5 * 2]/2;
% Unfortunately the python code can cross validate and update filter length
% based on specification. Assuming filter is going to be optimal and stays
% the same we can accept the length is known.

n = n + 1 * (mod(n,2)==0);
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

load sample_dat
g = grpdelay(b_py,1);
g = round(g(1));
ch = 1:4;
x = x*10^3;
y= filter(b,1,x,[],1);
y_py = filter(b_py,1,x,[],1);

figure()
subplot(3,1,1)
plot((1:length(x))/fs,x(:,ch),'linewidth',2)
xlim([0,2])
xlabel('time[s]')
ylabel('ampl [mV]')
subplot(3,1,2)
plot((1:length(y))/fs,y(:,ch),'linewidth',2)
xlim([0,2])
xlabel('time[s]')
ylabel('ampl [mV]')
subplot(3,1,3)
plot((1:length(y_py))/fs,y_py(:,ch),'linewidth',2)
xlim([0,2])
xlabel('time[s]')
ylabel('ampl [mV]')

figure()
subplot(2,1,1)
plot((1:length(y_py))/fs,y_py(:,ch)-y(:,ch),'linewidth',2)
xlim([0,2])
xlabel('time[s]')
ylabel('ampl [mV]')
