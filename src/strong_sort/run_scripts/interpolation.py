import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import json
import matplotlib.pyplot as plt


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class GSI:
    def __init__(self, data, N=None, kernel=None, multivariate=True):
        self.data = np.genfromtxt(data, delimiter=',', dtype=float)
        self.N = N
        self.kernel = kernel if kernel is not None else 1.0 * RBF()
        self.multivariate = multivariate
        self.max_frame = int(np.max(self.data[:, 0]))

        unique_track_ids = np.unique(self.data[:, 1])
        
        all_new_rows = []
        for track_id in unique_track_ids:
            new_rows = self.predict_missing_frames(track_id)
            if new_rows is not None:
                all_new_rows.append(new_rows)

        if all_new_rows:
            all_new_rows = np.concatenate(all_new_rows, axis=0)  # Use NumPy's concatenate instead of vstack for efficiency
            self.data = np.vstack((self.data, all_new_rows))
            self.data = self.data[np.argsort(self.data[:, 0])]

    def get_track_data(self, track_id):
        return self.data[self.data[:, 1] == track_id]

    def predict_missing_frames(self, track_id):
        track_data = self.get_track_data(track_id)
        
#        if np.all(track_data[0, 2:6] == track_data[:, 2:6]):
#            return None

        min_frame = int(np.min(track_data[:, 0]))

        missing_frames = np.setdiff1d(np.arange(min_frame, self.max_frame + 1), track_data[:, 0].astype(int))

        repeated_frames = track_data[np.where((track_data[:-1, 2:6] == track_data[1:, 2:6]).all(axis=1))][:, 0]
        
        all_missing_frames = np.unique(np.concatenate([missing_frames, repeated_frames]))

        if all_missing_frames.size == 0:
            return None

        X_missing = all_missing_frames[:, np.newaxis]

        new_rows = np.zeros((len(all_missing_frames), self.data.shape[1]))
        new_rows[:, 0] = all_missing_frames
        new_rows[:, 1] = track_id
        new_rows[:, 6] = 0.5
        new_rows[:, 7:] = -1
        new_rows[:, 8] = 1

        if self.N is not None:
            X = track_data[-self.N:, 0][:, np.newaxis]
            y = track_data[-self.N:, 2:6]
        else:
            X = track_data[:, 0][:, np.newaxis]
            y = track_data[:, 2:6]

        if self.multivariate:
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
            gp.fit(X, y)
            y_pred = gp.predict(X_missing)
            print(y_pred)
            new_rows[:, 2:6] = y_pred
        else:
            for feature_idx in range(2, 6):
                gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
                gp.fit(X, y[:, feature_idx - 2])
                y_pred = gp.predict(X_missing)
                print(y_pred)
                new_rows[:, feature_idx] = y_pred

        return new_rows

    def save_to_text_file(self, path):
        fmt = '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d'
        np.savetxt(path, self.data, fmt=fmt)

def plot_track_data(gsi, track_id, feature_idx):
    track_data = gsi.get_track_data(track_id)
    plt.figure()
    plt.title(f"Feature {feature_idx} for Track ID {track_id}")
    plt.plot(track_data[:, 0], track_data[:, feature_idx], 'o-', label='Interpolated')
    plt.xlabel('Frame')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    gsi = GSI(data='/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/video_16/tracker.txt')
    
    # Save the extrapolated data to a file
    gsi.save_to_text_file('/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/video_16/tracker_extrapolated.txt' )
    
    # Retrieve unique track IDs
    unique_track_ids = np.unique(gsi.data[:, 1])
    
    # Loop over each track ID and plot the features
    for track_id in unique_track_ids:
        track_data = gsi.get_track_data(track_id)
        for feature_idx in [2, 3, 4, 5]:  # For x, y, w, h respectively
            plt.figure()
            # Plot the line using all points
            plt.plot(track_data[:, 0], track_data[:, feature_idx], 'k-', label='Data')
            # Identify the interpolated points
            interpolated = (track_data[:, 6] == 0.5)  # Change this depending on how you've tagged your interpolated data
            # Scatter plot for the original data points (not interpolated)
            plt.scatter(track_data[~interpolated, 0], track_data[~interpolated, feature_idx], c='b', label='Original')
            # Scatter plot for interpolated data points
            plt.scatter(track_data[interpolated, 0], track_data[interpolated, feature_idx], c='r', label='Interpolated')
            plt.xlabel('Frame')
            plt.ylabel(f'Feature {feature_idx}')
            plt.legend()
            plt.title(f'Feature {feature_idx} for Track ID {track_id}')
            plt.show()

'''
class SimpleGaussianInterpolation:
    def __init__(self, data):
        self.data = np.genfromtxt(data, delimiter=',', dtype=float)  # Load data from file
        self.max_frame = int(np.max(self.data[:, 0]))
        unique_track_ids = np.unique(self.data[:, 1])
        
        for track_id in unique_track_ids:
            self.predict_missing_frames(track_id)

    def get_track_data(self, track_id):
        return self.data[self.data[:, 1] == track_id]

    def predict_missing_frames(self, track_id):
        track_data = self.get_track_data(track_id)
        min_frame = int(np.min(track_data[:, 0]))
        existing_frames = set(map(int, track_data[:, 0]))
        all_frames = set(range(min_frame, self.max_frame + 1))
        missing_frames = np.array(sorted(list(all_frames - existing_frames)))

        if len(missing_frames) == 0:
            return

        new_rows = np.zeros((len(missing_frames), self.data.shape[1]))
        new_rows[:, 0] = missing_frames
        new_rows[:, 1] = track_id
        new_rows[:, 6] = 1
        new_rows[:, 7:] = -1

        for feature_idx in [2, 3, 4, 5]:
            existing_values = track_data[:, feature_idx]
            differences = np.diff(existing_values)
            std_dev = np.std(differences) if len(differences) > 1 else 0

            existing_frames_list = list(existing_frames)
            for missing_frame in missing_frames:
                before_frame = max([f for f in existing_frames_list if f < missing_frame], default=None)
                after_frame = min([f for f in existing_frames_list if f > missing_frame], default=None)

                if before_frame is None or after_frame is None:
                    continue

                before_value = track_data[track_data[:, 0] == before_frame, feature_idx][0]
                after_value = track_data[track_data[:, 0] == after_frame, feature_idx][0]

                weight = (missing_frame - before_frame) / (after_frame - before_frame)
                interpolated_value = before_value + weight * (after_value - before_value)
                new_value = interpolated_value + np.random.normal(0, std_dev)

                new_rows[new_rows[:, 0] == missing_frame, feature_idx] = new_value

        self.data = np.vstack((self.data, new_rows))
        self.data = self.data[np.argsort(self.data[:, 0])]

    def save_to_text_file(self, path):
        fmt = '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d'
        np.savetxt(path, self.data, fmt=fmt)
'''