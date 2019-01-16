import glob
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    origin_data = '/mnt/vanity/data'
    new_data = '/mnt/vanity/csv'
    files = glob.glob(os.path.join(origin_data, '*.npy'))
    for i, file in enumerate(files):
        df = pd.DataFrame(np.load(file)[0:50000, :])
        name, ext = os.path.splitext(os.path.basename(file))
        df.to_csv(os.path.join(origin_data, name + '.csv'), header=None, index=None)
        print('{:02f}%'.format((i + 1) / 345 * 100))
