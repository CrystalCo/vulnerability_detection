import matplotlib.pyplot as plt
import seaborn as sns

def plotHistogram(mydata, colName):
    metricStr = colName

    x1 = list(mydata[mydata[metricStr] == 'TN']['DLOutput'])
    x2 = list(mydata[mydata[metricStr]  == 'FP']['DLOutput'])
    x3 = list(mydata[mydata[metricStr]  == 'TP']['DLOutput'])
    x4 = list(mydata[mydata[metricStr] == 'FN']['DLOutput'])

    plt.ylim(0, 10)
    # Assign colors for each airline and the names
    colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73']
    names = ['TN', 'FP', 'TP', 'FN']
         
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist([x1, x2, x3, x4], bins = int(10), normed=True,
         color = colors, label=names)

    # Plot formatting
    plt.legend()
    plt.xlabel('DL Outputs')
    plt.ylabel('count')
    plt.title('Prediction Histogram (30,000 API)')

    count = mydata[metricStr].value_counts()
    print(count)

def plotBar(mydata, colName):
    value = mydata[colName]

    # matplotlib histogram
    plt.hist(value, color = 'blue', edgecolor = 'black',
         bins = 10)

    # Add labels
    plt.title('Metric Histogram')
    plt.xlabel('DL Value')
    plt.ylabel('count')
    count = value.value_counts()
    print(count)
