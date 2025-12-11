import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
import h5py

class annotateClass:
    '''A class to represent an annotation for time series data.
    
    Attributes:
        timeStart (float): A float representing the start time of the annotation in seconds.
        timeEnd (float): A float representing the end time of the annotation in seconds.
        text (str): A string representing the text of the annotation.
        validated (bool): A boolean indicating whether the annotation has been validated.
    '''

    def __init__(self, timeStart, timeEnd, label, validated = False):
        '''Initializes an annotateClass object.
        
        Args:
            timeStart (float): A float representing the start time of the annotation in seconds.
            timeEnd (float): A float representing the end time of the annotation in seconds.
            text (str): A string representing the text of the annotation.
            validated (bool): Optional; a boolean indicating whether the annotation has been validated. Defaults to False.
        '''
        self.timeStart = timeStart  # in seconds
        self.timeEnd = timeEnd  # in seconds
        self.label = label
        self.validated = validated

def load_Annotation(filePath):
    '''Loads annotations from a CSV file and returns a list of annotateClass objects.
    
    Args:
        filePath (str): A string representing the path to the CSV file containing the annotations.

    Returns:
        list: A list of annotateClass objects representing the annotations.
    '''

    timeStart = []
    timeEnd = []
    label = []
    validated = []

    df = pd.read_csv(filePath)
    df.columns = df.columns.str.replace(r'\s+', '', regex=True)

    for index, row in df.iterrows():
        tmp = pd.to_datetime(row['TimeStart'].strip()).time()
        timeStart.append(tmp.hour*3600 + tmp.minute*60 + tmp.second)
        tmp = pd.to_datetime(row['TimeEnd'].strip()).time()
        timeEnd.append(tmp.hour*3600 + tmp.minute*60 + tmp.second)
        label.append(row['Label'].strip())
        validated.append(row.get('validated', False))
    
    ann = annotateClass(timeStart,timeEnd,label,validated)
    return ann

class dataClass:
    '''A class to represent a data container for time series data.
    
    Attributes:
        time (Series): A pandas Series object representing time of the day in seconds when the data was recorded.
        data (DataFrame): A pandas DataFrame containing the data to be stored.
        time_range (list): A list containing the earliest and latest timestamps from the time Series.
        date_recording (datetime): A datetime object representing the date of recording.
        dt_sampling (float): A float representing the sampling interval in seconds.'''
    
    def __init__(self, time, data, data_description, date_recording, dt_sampling = -1.0):
        '''Initializes a dataClass object.
        
        Args:
            time (Series): A pandas Series object representing time of the day in seconds when the data was
                recorded.
            data (DataFrame): A pandas DataFrame containing the data to be stored.
            data_description (str): A string describing the data.
            date_recording (datetime): A datetime object representing the date of recording.
            dt_sampling (float): Optional; a float representing the sampling interval in seconds.
        '''
        self.time = time # in seconds
        self.data = data
        self.data_description = data_description
        self.time_range = timeRange(time)
        self.date_recording = date_recording
        self.dt_sampling = dt_sampling # in seconds

    def plotData(self,downsample = 1):
        '''Plots the data in a line plot with markers using Plotly.
        
        Args:
            downsample (int): Optional; an integer specifying the downsampling factor for the data.
        '''
        fig = go.Figure()
        plotData(fig,self.time,self.data,downsample)
        fig.show()

def plotData(fig,t,data,downsample=1):
    '''Generates and displays an interactive line plot using Plotly.

    Args:
        fig (Figure): A Plotly Figure object.
        t (Series): A pandas Series representing the time values for the x-axis.
        data (DataFrame): A pandas DataFrame representing the data values for the y-axis.
        downsample (int): Optional; an integer specifying the downsampling factor for the data.
    '''

    t = pd.to_timedelta(t[::downsample], unit="s") # Converting from seconds to hours
    origin = pd.Timestamp("1970-01-01") # Add artificial date
    t_dt = origin + t # New datetime timestamp for better visualization

    for column in data.columns:
        val = data[column][::downsample]
        fig.add_trace(go.Scatter(x=t_dt, y=val, mode='lines', name=column))

    fig.update_layout(
        title = "Data Over Time",
        xaxis_title = "Time (hours)",
        yaxis_title = "Values",
        xaxis = dict(
            showgrid = True,
            tickangle = 0,
            tickformat="%H:%M:%S.%f",
            nticks = 10,
            rangeselector = dict(
                buttons = list([
                    dict(count=5, label="5s", step="second", stepmode="backward"),
                    dict(count=10, label="10s", step="second", stepmode="backward"),
                    dict(count=30, label="30s", step="second", stepmode="backward"),
                    dict(count=1, label="1m", step="minute", stepmode="backward"),
                    dict(step="all")
                ])
            )
            #type="date"
        ),
        yaxis = dict(showgrid=True),
        template = "plotly_white"
    )

def plotDataH5(fig,f,dataset_name, downsample=1):

    t = f[dataset_name]['time'][()]
    data = f[dataset_name]['data'][()]
    df = pd.DataFrame(data, columns=f[dataset_name]['data'].attrs.get('column_descriptions',None).astype(str))

    plotData(fig,t,df,downsample)

def load_BPMonitor(filePath):
    '''Loads blood pressure data from a CSV file and returns a dataClass object.
    
    Args:
        filePath (str): A string representing the path to the CSV file containing the data.
        
    Returns:
        A dataClass object containing the blood pressure data.
    '''
    # Loading the raw data
    content = pd.read_csv(filePath)

    # Extracting baseline date and date/times for samples
    date = content["Date"]
    time = content["Time"]
    data_DateFormat = "%b %d, %Y %I:%M %p"
    date_recording = combineDateTime(date[0],"12:00 AM", data_DateFormat)
    datetimes = combineDateTime(date, time, data_DateFormat)

    # Converting samples from date/time to offset seconds
    dt = datetimes-date_recording
    dt = dt.dt.total_seconds()
    dt.name = "Time(sec)"

    # Computing range
    range_time = timeRange(dt)

    return dataClass(dt, content[['SYS(mmHg)','DIA(mmHg)','Pulse(Beats/Min)']],
                     "Blood pressure data.", date_recording)


def load_VideoRange(filePath):
    '''Loads video range data from a CSV file and returns a dataClass object.
    
    Args:
        filePath (str): A string representing the path to the CSV file containing the data.
    
    Returns:
        A dataClass object containing the video range data.
    '''

    # Loading the raw data
    content = pd.read_csv(filePath)

    # Extracting baseline date and date/times for samples
    date = content["Date"]
    time = content["Time"]
    data_DateFormat = "%b %d, %Y %I:%M:%S %p"
    date_recording = combineDateTime(date[0],"12:00:00 AM", data_DateFormat)
    datetimes = combineDateTime(date, time, data_DateFormat)

    # Converting samples from date/time to offset seconds
    dt = datetimes-date_recording
    dt = dt.dt.total_seconds()
    dt.name = "Time(sec)"

    # Computing range
    range_time = timeRange(dt)

    return dataClass(dt, content[['Value']],
                     "Video range data.", date_recording)


def load_BiopacECG(filePath):
    '''Loads ECG data from a CSV file and returns a dataClass object.
    
    Args:
        filePath (str): A string representing the path to the CSV file containing the data.
        
    Returns:
        A dataClass object containing the blood pressure data.
    '''
    # Loading the raw data
    content = pd.read_csv(filePath, sep='\t', names=["sec", "CH2", "None"], skiprows=9)

    # Extracting ecg values
    ecg = content['CH2']
    ecg.name = 'ECG(mV)'
    ecg = ecg.to_frame()

    # Extracing date/time of recording from file
    with open(filePath, 'r') as file:
        file.readline()
        file.readline()
        startingTime = file.readline()
    startingTime = startingTime.split()

    # Converting date/time of recording to datetime
    date = startingTime[2]
    time = startingTime[3]
    data_DateFormat = "%Y-%m-%d %H:%M:%S.%f"
    datetime_recording = combineDateTime(date, time, data_DateFormat)

    # Extracting date of recording
    date_recording = combineDateTime(date,"00:00:00.00", data_DateFormat)

    # Specifying sampling rate
    dt_sampling = 1.0/2000  # 2000 Hz sampling rate

    # Extracting time values
    #dt = np.array(range(len(ecg)))*dt_sampling
    #dt = dt + (datetime_recording - date_recording).total_seconds()
    #dt = pd.Series(dt, name="Time(sec)")
    dt = content['sec']
    dt.name = "Time(sec)"
    dt = dt + (datetime_recording - date_recording).total_seconds()

    # Computing range
    range_time = timeRange(dt)

    return dataClass(dt, ecg, "ECG data.", date_recording, dt_sampling)

def load_IMUAccel(filePath):
    '''
    Loads accelerometer data from an IMU CSV file and returns a dataClass object.

    Args:
        filePath (str): A string representing the path to the CSV file containing the IMU accelerometer data.

    Returns:
        dataClass: A dataClass object containing the processed IMU accelerometer data and timestamps.
    '''
    content = pd.read_csv(filePath)
    
    #extract IMU values
    time = content["timestamp (-0400)"]
    x = content["x-axis (g)"]
    y = content["y-axis (g)"]
    z = content["z-axis (g)"]
    epoch_ms = content["epoc (ms)"]

    #extract dates and times
    time = time.str.split("T", expand = True)
    dates = time[0]
    times = time[1]

    combinedFormat = "%Y-%m-%d %H.%M.%S.%f"
    datetimes = combineDateTime(dates, times, combinedFormat)

    #starting date and time
    datetime_recording = datetimes[0]

    #extract only the start date
    date_recording = combineDateTime(dates[0],"00.00.00.00", combinedFormat)

    dt_sampling = 1/100

    epoch_offset_sec = (epoch_ms - epoch_ms.iloc[0]) / 1000.0
    base_offset_sec = (datetime_recording - date_recording).total_seconds()
    dt = epoch_offset_sec + base_offset_sec
    dt = pd.Series(dt, name="Time(sec)")

    return dataClass(dt, content[["x-axis (g)", "y-axis (g)", "z-axis (g)"]],
                        "Accelerometerer data.", date_recording, dt_sampling)

def load_IMUGyro(filePath):
    '''
    Loads gyroscope data from an IMU CSV file and returns a dataClass object.

    Args:
        filePath (str): A string representing the path to the CSV file containing the IMU gyroscope data.

    Returns:
        dataClass: A dataClass object containing the processed IMU gyroscope data and timestamps.
    '''

    content = pd.read_csv(filePath)
    
    #extract IMU values
    time = content["timestamp (-0400)"]
    x = content["x-axis (deg/s)"]
    y = content["y-axis (deg/s)"]
    z = content["z-axis (deg/s)"]

    epoch_ms = content["epoc (ms)"]

    #extract dates and times
    time = time.str.split("T", expand = True)
    dates = time[0]
    times = time[1]

    combinedFormat = "%Y-%m-%d %H.%M.%S.%f"
    datetimes = combineDateTime(dates, times, combinedFormat)

    #starting date and time
    datetime_recording = datetimes[0]

    #extract only the start date
    date_recording = combineDateTime(dates[0],"00.00.00.00", combinedFormat)

    # Convert 'epoc (ms)' to datetime
    epoch_ms = content["epoc (ms)"]
    datetime_series = pd.to_datetime(epoch_ms, unit='ms')

    # Calculate seconds offset from the start datetime
    epoch_offset_sec = (epoch_ms - epoch_ms.iloc[0]) / 1000.0
    base_offset_sec = (datetime_recording - date_recording).total_seconds()
    dt = epoch_offset_sec + base_offset_sec
    dt = pd.Series(dt, name="Time(sec)")


    dt_sampling = 1/100

    return dataClass(dt, content[['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']],
                        "Gyroscope data.", date_recording, dt_sampling)

def load_EDA(filePath):
    '''
    Loads EDA data from an HDF5 file and returns a dataClass object.

    Args:
        filePath (str): A string representing the path to the HDF5 file containing the EDA data.

    Returns:
        dataClass: A dataClass object containing the processed EDA data and timestamps.
    '''
    with h5py.File(filePath, 'r') as file:

        #firstKey is: ['00:21:08:35:18:B7']
        firstKey = list(file.keys())[0]
        firstKey = file[firstKey]

        #secondKey is: ['digital', 'events', 'plugin', 'raw', 'support']
        secondKey = list(firstKey.keys())

        #access just "raw"
        #keys inside of "raw": ['channel_1', 'nSeq']
        rawKey = firstKey[secondKey[3]]

        channel_1_data = rawKey['channel_1'][:].squeeze()

        samplingRate = firstKey.attrs['sampling rate']
        samplingRate = 1/samplingRate

        startTime = firstKey.attrs['time']
        startDate = firstKey.attrs['date']

        data_DateFormat = "%Y-%m-%d %H:%M:%S.%f"
        datetime_recording = combineDateTime(startDate, startTime, data_DateFormat)
        date_recording = combineDateTime(startDate,"00:00:00.00", data_DateFormat)


        dt = np.array(range(0,len(channel_1_data)))*samplingRate + (datetime_recording - date_recording).total_seconds()
        dt = pd.Series(dt, name="Time(sec)")

        eda_df = pd.DataFrame({'EDA': channel_1_data})


    return dataClass(dt, eda_df, "EDA data.", date_recording, samplingRate)


def combineDateTime(dates, times, dataFormat):
    '''Combines a date series and a time series into one series of datetime objects.

    Takes in a date series and a time series and combines them into one series. once
    combined, it utilizes "pd.to_datetime()" to convert each row into a datetime object. 

    Args:
        dates: A pandas Series containing date strings.
        times: A pandas Series containing time strings.
        dataFormat: A format string used to parse the combined date-time strings.

    Returns:
        A pandas Series containing datetime objects representing the combined
        date and time. The resulting Series is named "Time".
    '''
    combined = dates + " " + times
    combined = pd.to_datetime(combined, format = dataFormat)
    combined.name = "Time"
    return combined

def formatDate(combined, desiredFormat):
    '''Formats a series of datetime objects into desired format.

    Converts a pandas Series of datetime objects into string
    representations using the specified output format.

    Args:
        combined: A pandas Series containing datetime objects.
        desiredFormat: A string specifying the desired output format.

    Returns:
        A pandas Series of formatted datetime strings.
    '''
    formattedCombined = combined.dt.strftime(desiredFormat)
    return formattedCombined

def timeRange(datetimes):
    '''Calculates the earliest and latest timestamps from a Series of datetime values.

    Identifies the minimum (earliest) and maximum (latest) datetime values in the provided
    pandas Series and returns them in a dictionary.

    Args:
        datetimes: A pandas Series containing datetime objects.

    Returns:
        A list of two elements: the earliest and latest datetime values.
    '''
    earliest = float(datetimes.min())
    latest = float(datetimes.max())
    return [earliest, latest]

def openData(filePath):
    '''Opens and reads the contents of a file.

    Reads the entire content of the file at the specified path and
    returns it as a string.

    Args:
        filePath: A string representing the path to the file.

    Returns:
        A string containing the contents of the file.
    '''
    with open(filePath, 'r') as file:
        content = file.read()
    return content


def makePlot(x, y):
    '''Generates and displays a line plot with markers using the provided x and y data.

    This function reverses the order of both x and y Series to plot them from the end to the beginning,
    then creates a line plot with circular markers and dashed lines. It automatically labels the axes
    based on the names of the Series and displays a grid and axis ticks for readability.

    Args:
        x: A pandas Series representing the values for the x-axis. Should have a `.name` attribute for labeling.
        y: A pandas Series representing the values for the y-axis. Should have a `.name` attribute for labeling.
    '''
    x = x.loc[::-1]
    y = y.loc[::-1]

    xtitle = str(x.name)
    ytitle = str(y.name)

    plt.plot(x, y, c="black",marker="o",ms=10,ls="--")
    plt.ylabel(ytitle)
    plt.title(ytitle + " Over " + xtitle)
    plt.xlabel(xtitle)
    plt.grid(True)
    plt.tick_params(width=2)
    plt.show()

def _makeTimePlot(ax, labels, starts, ends, widths, y_pos, time_starts, time_ends, labels_list):
    
    ax.barh(y_pos, widths, left=starts, height=0.4, color='C0', alpha=0.8)

    # Helper to convert seconds-since-midnight to HH:MM string
    def sec_to_hhmm(sec):
        dt = (datetime.datetime.combine(datetime.date.today(), datetime.time(0)) +
            datetime.timedelta(seconds=float(sec)))
        return dt.strftime('%H:%M:%S')

    # Annotate each bar with its time range
    for yi, s, e in zip(y_pos, starts, ends):
        ax.text(s + 0.02*(max(ends)-min(starts)), yi,
                f"{sec_to_hhmm(s)} → {sec_to_hhmm(e)}",
                va='center', ha='left', color='white' if widths[y_pos.index(yi)]>0 else 'black',
                fontsize=9, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (HH:MM:SS, seconds since midnight)')
    ax.set_title('Data Recording Time Ranges')

    # Create sensible x-ticks every 15 minutes (900s) within the domain
    min_s = int(min(starts))
    max_e = int(max(ends))
    tick_step = 900  # 15 minutes
    tick_start = (min_s // tick_step) * tick_step
    tick_end = ((max_e + tick_step - 1) // tick_step) * tick_step
    ticks = list(range(tick_start, tick_end + 1, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sec_to_hhmm(t) for t in ticks], rotation=45, ha='right')

    ax.set_xlim(min_s - 60, max_e + 60)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if time_starts is None or time_ends is None or labels_list is None:
        raise AttributeError("Could not find annotation start/end/label arrays on `ann`. "
                            "Expected attributes like timeStart/timeEnd/label or ann.df with those columns.")

    # Ensure sequences are plain lists (handle pandas Series)
    time_starts = list(time_starts)
    time_ends = list(time_ends)
    labels_list = list(labels_list)

    # Build color map for unique labels (preserve order)
    unique_labels = list(dict.fromkeys(labels_list))
    cmap = plt.get_cmap('tab10')
    color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}

    # Plot vertical shaded regions across the whole y-range
    for s, e, lab in zip(time_starts, time_ends, labels_list):
        ax.axvspan(float(s), float(e), color=color_map[lab], alpha=0.25, zorder=0)

    # Add legend with color swatches
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[l], alpha=0.25) for l in unique_labels]
    ax.legend(handles, unique_labels, title='Annotations', bbox_to_anchor=(1.02, 1), loc='upper left')


def makeTimePlot(ax,data,ann):
    '''Generates a horizontal timeline bar plot for given time ranges and overlays annotation regions.
    
    This function creates a horizontal bar plot to visualize the recording time ranges of various sensors.
    It also overlays shaded vertical regions to represent annotations, with colors corresponding to different labels.

    Args:
        timeRanges (dict): A dictionary where keys are sensor names and values are lists of [start_time, end_time] in seconds.
        ann (annotateClass): An annotateClass object containing annotation start times, end times, and labels.
    '''
    # Extracting time range data for plotting
    labels = list(data.keys())
    starts = [data[l].time_range[0] for l in labels]   # seconds since midnight
    ends   = [data[l].time_range[1] for l in labels] 
    widths = [e - s for s, e in zip(starts, ends)]
    y_pos = list(range(len(labels)))[::-1]  # reverse so first label appears at top

    # Getting start and end times from annotations
    time_starts = ann.timeStart
    time_ends   = ann.timeEnd
    labels_list = ann.label

    # Create the base time plot with bars
    _makeTimePlot(ax, labels, starts, ends, widths, y_pos, time_starts, time_ends, labels_list)

    # Plotting markers for 
    for index, l in enumerate(labels):
        if(data[l].dt_sampling < 0):
            t = data[l].time
            ax.plot(t,y_pos[index]*np.ones(len(t)),'ks',markersize=10,alpha=0.5)

    plt.show()


def makeTimePlotH5(ax,f):
    '''Generates a horizontal timeline bar plot for given time ranges and overlays annotation regions.
    
    This function creates a horizontal bar plot to visualize the recording time ranges of various sensors.
    It also overlays shaded vertical regions to represent annotations, with colors corresponding to different labels.
        
    Args:
        f (h5py.File): An open HDF5 file object containing time range data and annotations.
    '''

    # Extracting time range data for plotting
    labels = sorted(list(set(f.keys()).difference({'Annotations'})))
    starts = [f[l]['time'][()].min() for l in labels]   # seconds since midnight
    ends   = [f[l]['time'][()].max() for l in labels] 
    widths = [e - s for s, e in zip(starts, ends)]
    y_pos = list(range(len(labels)))[::-1]  # reverse so first label appears at top

    # Getting start and end times from annotations
    tmp = f['Annotations']['time'][()]
    time_starts = [ran[0] for ran in tmp]
    time_ends   = [ran[1] for ran in tmp]
    labels_list = f['Annotations']['data'][()]
    labels_list = [item.decode('utf-8') for item in labels_list]

    # Create the base time plot with bars
    _makeTimePlot(ax, labels, starts, ends, widths, y_pos, time_starts, time_ends, labels_list)

    # Plotting markers for 
    for index, l in enumerate(labels):
        if(f[l]['sampling_rate'][()] < 0):
            t = f[l]['time'][()]
            ax.plot(t,y_pos[index]*np.ones(len(t)),'ks',markersize=10,alpha=0.5)
