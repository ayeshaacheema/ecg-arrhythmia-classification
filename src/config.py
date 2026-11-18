# ===== Shared project settings: edit here only =====

PROJECT   = '/content/drive/MyDrive/ECG_Arrhythmia_Classification'
RAW_DIR   = f'{PROJECT}/data/raw/mitdb'
PROC_DIR  = f'{PROJECT}/data/processed'
FIG_DIR   = f'{PROJECT}/figures'
MODEL_DIR = f'{PROJECT}/models'
RES_DIR   = f'{PROJECT}/results'

SEED = 42          # same randomness every run -> reproducible
LEAD = 'MLII'      # always select lead by name (record 114 has swapped leads)

# Beat window around R-peak (360 Hz)
FS = 360
BEFORE = 90        # 0.25 s before R-peak
AFTER  = 162       # 0.45 s after R-peak

# Patient-disjoint split (de Chazal et al., 2004), record 202 removed from test
DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,
       201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,210,212,213,
       214,219,221,222,228,231,232,233,234]
PACED = [102,104,107,217]
VAL_RECORDS   = [118, 124, 205, 215, 220]           # chosen in 01_data_exploration (mix of S and V)
TRAIN_RECORDS = [r for r in DS1 if r not in VAL_RECORDS]   # 17 patients
TEST_RECORDS  = DS2                                  # 21 patients, used ONLY for final evaluation

# MIT-BIH symbol -> AAMI class
AAMI = {'N':'N','L':'N','R':'N','e':'N','j':'N',
        'A':'S','a':'S','J':'S','S':'S',
        'V':'V','E':'V',
        'F':'F',
        '/':'Q','f':'Q','Q':'Q'}

# Binary label: Normal=0, Abnormal=1, Q excluded
BINARY = {'N':0, 'S':1, 'V':1, 'F':1}
RR_COLS = ['pre_rr', 'post_rr', 'local_rr', 'long_rr',
           'rel_pre_local', 'rel_post_local', 'rel_pre_long', 'rel_post_long']
