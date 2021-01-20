## How to use text_filter() function
### Input parameters:

This function processes the raw EEG input through a bandpass filter. Three default filters are hard-coded and can be chosen by specifying the sampling freqeuency of the hardware. Three filters are designed for 256Hz, 300Hz and 1024Hz sampling rates. If another filter is required to be used, it can be passed to the function. Input parameters are:

* ```input_inq```

This parameter is the input multi channel EEG signal. Expected dimensions are Number of Channels x Number of Samples

* ```filt```

Input for using a specific filter. If left empty, according to sampling frequency, a pre-designed filter is going to be used. Filters are pre-designed for fs = 256, 300 or 1024 Hz. For sampling frequencies besides these values, filter needs to be provided to the function.

* ```fs```

Sampling frequency of the hardware in Hz. Default value = 256

* ```k```

Downsampling order. Default value = 2
### Usage:

Pass an input eeg np.array that is a matrix where every row is a channels data. For example a two channel EEG inquiry could be:

```python
input_inq = np.array([[1, 4, ...],
       	               [2, 2, ...]])
```

Specify parameters. If your sampling frequency is different than predefined values, specify the filter. Returned value is another numpy array in the form:

```python
output_inq = np.array([[.3, .4, ...],
       	                [.2, .1, ...]])
```

For other details, refer to demo file or function definition.