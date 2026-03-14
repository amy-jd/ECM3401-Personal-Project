CONTEXT_WINDOW = 382 #4 days
FORECAST_WINDOW = 0 #48 # 12 hours
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

WEATHER_COLS = ['rainfall']
TEMPORAL_COLS = ['part_of_day', 'part_of_week', 'part_of_year']
TEMPORAL_EMBEDDING_DIMENSIONS = [6, 2, 4]

LEARNING_RATE = 0.0005
NB_EPOCHS = 35
BATCH_SIZE = 32
EARLY_STOPPING_THRESHOLD = 30
MODEL_SAVE_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Code\test\ECM3401-Personal-Project\project\best_model.pth"

EMBED_DIM = 32

GNN_DROPOUT = 0.01

FLOWDATA_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\FlowData_for_SubsetGraph_12_with_7SensoredPipes.csv"
SUBSETGRAPH_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\SubsetGraph_12_with_7SensoredPipes.csv"
WEATHERDATA_PATH = r"C:\Users\ameli\Documents\Uni\year-3-notes\diss\Dataset\rainfall_hadukgrid_uk_region_day_18910101-20241231.nc"

TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]


STRATA_TO_INDEX = {
    "part_of_day": {
        "night": 0,
        "early-morning": 1,
        "late-morning": 2,
        "afternoon": 3,
        "early-evening": 4,
        "late-evening": 5
    },
    "part_of_week": {
        "weekday": 0,
        "weekend": 1
    },

    "part_of_year": {
        "winter": 0,
        "spring": 1,
        "summer": 2,
        "autumn": 3
    }
}
