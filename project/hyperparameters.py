CONTEXT_WINDOW = 96
FORECAST_WINDOW = 24
TOTAL_WINDOW = CONTEXT_WINDOW + FORECAST_WINDOW
MAX_GAP = 16

#SENSOR_COLS = ['919', '157', '1959', '1016', '1994', '1870']
SENSOR_COLS = [0,1,2,3,4,5]
SENSOR_DMA_TO_ID = {
    '919': 0, 
    '157': 1, 
    '1016': 2, 
    '1870': 3,
    '1959': 4, 
    '1994': 5,
}

LEARNING_RATE = 0.005
NB_EPOCHS = 30
BATCH_SIZE = 16

FLOWDATA_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\FlowData_for_SubsetGraph_12_with_7SensoredPipes.csv"
SUBSETGRAPH_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\SubsetGraph_12_with_7SensoredPipes.csv"

