CONTEXT_WINDOW = 382 #4 days
FORECAST_WINDOW = 24 # 12 hours
TOTAL_WINDOW = CONTEXT_WINDOW + FORECAST_WINDOW # 4.5 days
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

LEARNING_RATE = 0.001
NB_EPOCHS = 25
BATCH_SIZE = 16

FLOWDATA_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\FlowData_for_SubsetGraph_12_with_7SensoredPipes.csv"
SUBSETGRAPH_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\SubsetGraph_12_with_7SensoredPipes.csv"

TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]

