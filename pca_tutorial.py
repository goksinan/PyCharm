
# THIS PROGRAM INTENDS TO REVEAL THE "BEHIND THE SCENES" OF THE PRINCIPAL COMPONENT ANALYSIS PROCEDURE
# A 3-DIMENSIONAL DATA SET WILL BE USED
# IN OTHER WORDS, THERE WILL BE 3 MEASUREMENT TYPES AND A NUMBER OF OBSERVATIONS
# PUTTING IN A MATRIX, COLUMNS CAN BE CONSIDERED AS VARIABLES(DIMENSIONS) AND ROWS AS TIME SAMPLES

## IMPORT
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

## START WITH GENERATING 3 SIGNALS
signal = np.linspace(0,20,21)  # The backbone of all 3 signals will be the same

x1 = signal + 3 + 4*np.floor(np.random.randn(len(signal)))
x2 = signal + 4 + 4*np.floor(np.random.randn(len(signal)))
x3 = signal + 5 + 4*np.floor(np.random.randn(len(signal)))

## PLOT DATA POINTS IN 3-D. ALSO ADD BASIS VECTORS
# Create a trace
trace = go.Scatter(
    x = x1,
    y = x2,
    mode = 'markers'
)

data = [trace]

py.iplot(data, filename='basic-scatter')