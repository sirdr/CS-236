import numpy as np
from sklearn.manifold import TSNE
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA


# MFCC
def mfcc_distance(x, y, sr=8000, outdir='', experiment=None):
    title = r'MFCCs & MSE'
    if experiment is not None:
        if 'Experiment' in experiment or 'experiment' in experiment:
            title += ' | {}'.format(experiment)
        else:
            title += ' | Experiment {}'.format(experiment)

    num_samples = len(x)
    mfccs_x = []
    mfccs_y = []
    print("Collecting MFCCs of inputs and predictions...")
    for i in tqdm(range(num_samples)):
        mfccs_x.append(np.expand_dims(librosa.feature.mfcc(y=x[i],sr=sr), axis=0))
        mfccs_y.append(np.expand_dims(librosa.feature.mfcc(y=y[i],sr=sr), axis=0))
    print("done")

    mfccs_x = np.vstack(mfccs_x)[:, 1:, :] # remove first mfcc
    mfccs_y = np.vstack(mfccs_y)[:, 1:, :]

    mfcc_distance = (np.square(mfccs_x - mfccs_y)).mean(axis=-1)

    mfcc_distance_reduced = mfcc_distance.mean(axis=-1)
    approx_min_ind, approx_max_ind = np.argmin(mfcc_distance_reduced, axis=0), np.argmax(mfcc_distance_reduced, axis=0)
    min_val , max_val =np.amin(mfcc_distance), np.amax(mfcc_distance)
    print("{} | {}".format(approx_min_ind, approx_max_ind))

    mfcc_distance_mean = mfcc_distance.mean(axis=0)
    mfcc_distance_std = mfcc_distance.std(axis=0)
    # mfcc_distance_stats['means'] = mfcc_distance_mean
    # mfcc_distance_stats['std'] = mfcc_distance_std

    # fig, axs = plt.subplots(ncols=3, sharey=True)
    # axs[0].imshow(np.expand_dims(mfcc_distance[approx_min_ind], axis=1), interpolation="none", cmap="plasma", aspect="auto", vmin=min_val, vmax=max_val, extent=[0,1,4000,0])
    # axs[1].imshow(np.expand_dims(mfcc_distance[approx_max_ind], axis=1), interpolation="none", cmap="plasma", aspect="auto",vmin=min_val, vmax=max_val, extent=[0,1,4000,0])
    # im = axs[2].imshow(np.expand_dims(mfcc_distance_mean, axis=1), interpolation="none", cmap="plasma", aspect="auto",vmin=min_val, vmax=max_val, extent=[0,1,4000,0])

    # axs[0].set_xlabel('Min')
    # axs[1].set_xlabel('Max')
    # axs[2].set_xlabel('Mean')

    fig=plt.figure(figsize=(14, 5))
    plt.subplot(1, 7, 1)
    ax=librosa.display.specshow(np.expand_dims(mfcc_distance[approx_min_ind], axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax.set_xlabel(r"$MSE^{\min}$")
    plt.subplot(1, 7, 2)
    ax1=librosa.display.specshow(np.expand_dims(mfcc_distance[approx_max_ind], axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax1.set_xlabel(r"$MSE^{\max}$")
    plt.subplot(1, 7, 3)
    ax2=librosa.display.specshow(np.expand_dims(mfcc_distance_mean, axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax2.set_xlabel(r"$MSE^{mean}$")
    cb1 = plt.colorbar()
    cb1.set_label('MSE')
    plt.subplot(1, 7, 4)
    ax3=librosa.display.specshow(np.expand_dims(mfccs_x[approx_max_ind, :, 6], axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax3.set_xlabel(r"$MFCC_{input}^{\max{MSE}}$")
    plt.subplot(1, 7, 5)
    ax4=librosa.display.specshow(np.expand_dims(mfccs_y[approx_max_ind, :, 6], axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax4.set_xlabel(r"$MFCC_{pred}^{\max{MSE}}$")
    plt.subplot(1, 7, 6)
    ax5=librosa.display.specshow(np.expand_dims(mfccs_x[approx_min_ind, :, 6], axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax5.set_xlabel(r"$MFCC_{input}^{\min{MSE}}$")
    plt.subplot(1, 7, 7)
    ax6=librosa.display.specshow(np.expand_dims(mfccs_y[approx_min_ind, :, 6], axis=1),y_axis='mel', sr=sr, fmax=sr/2)
    ax6.set_xlabel(r"$MFCC_{pred}^{\min{MSE}}$")

    image_path = os.path.join(outdir, 'mfcc-distances.png')
    
    cb2 = plt.colorbar()
    cb2.set_label('Power')
    plt.suptitle(title)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(image_path)

    return approx_min_ind, approx_max_ind, mfcc_distance_mean.mean(), mfcc_distance_std.mean()


def visualize_embeddings(encodings, outdir='', experiment=None):
    title = r'PCA of Latent Encodings'
    if experiment is not None:
        if 'Experiment' in experiment or 'experiment' in experiment:
            title += ' | {}'.format(experiment)
        else:
            title += ' | Experiment {}'.format(experiment)
    n_samples = len(encodings)
    X = np.reshape(encodings, (n_samples, -1))
    pca = PCA(n_components=2)
    X_fit = pca.fit_transform(X)
    image_path = os.path.join(outdir, 'latent-viz.png')
    plt.figure()
    plt.scatter(X_fit[:,0], X_fit[:,1])
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(image_path)

def avg_losses(losses_dict_list, outdir=''):

    losses = {}
    for ld in losses_dict_list:
        for k, v in ld.items():
            if losses.get(k) is None:
                losses[k] = [v]
            else:
                losses[k].append(v)

    avg_losses_dict = {l: np.mean(v) for l, v, in losses.items()}
    avg_losses_dict_str = str(avg_losses_dict)
    path = os.path.join(outdir, 'avg-losses.txt')
    print("Writing average losses -- {} -- to {}".format(avg_losses_dict_str, path))
    with open(path, "w") as w:
        w.write(avg_losses_dict_str)

def gather_shards(dict_of_dicts, output=True, outdir=''):
    def collect_as_numpy(dictionary):
        array = []
        for k in sorted(dictionary):
            array.append(dictionary[k])
        return np.vstack(array)

    new_dict = {}
    for k, v in dict_of_dicts.items():
        n = collect_as_numpy(v)
        new_dict[k] = n
        if output:
            path = os.path.join(outdir, "val-all-{}.npy".format(k))
            print('writing all {} to {}...'.format(k, path))
            np.save(path, n)
    return new_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dir', type=str, help='path to eval directory')
    parser.add_argument('--experiment', type=str, help='')
    args = parser.parse_args()

    losses_dict_list = []
    names_to_data = {}

    for filename in os.listdir(args.eval_dir):
        path = os.path.join(args.eval_dir, filename)
        if "losses.pickle" in filename:
            with open(path, "r") as r:
                losses_dict_list.append(pickle.load(r))
        elif filename.endswith(".npy"):
            split_info= filename.split('.')[0].split('-')
            if split_info[1] == "all":
                continue
            split_number = int(split_info[1])
            split_name = split_info[-1]
            if names_to_data.get(split_name) is None:
                names_to_data[split_name]={split_number: np.load(path)}
            else:
                names_to_data[split_name][split_number] = np.load(path)

    print('gathering shards...')
    names_to_data = gather_shards(names_to_data, output=True, outdir=args.eval_dir)
    print('done')

    results_path = os.path.join(args.eval_dir, 'results')
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Gather loss statisitcs 
    avg_losses(losses_dict_list, outdir=args.eval_dir)

    # Create MFCC plots
    min_ind, max_ind, mfcc_mean, mfcc_std = mfcc_distance(names_to_data['audio'], names_to_data['predictions'], outdir=results_path, experiment=args.experiment)

    # Gather data Statistics
    audio_mean, audio_std = names_to_data['audio'].mean(), names_to_data['audio'].std()
    pred_mean, pred_std = names_to_data['predictions'].mean(), names_to_data['predictions'].std()

    statistics = { 
                    "mfcc" : {"mean": mfcc_mean, "std": mfcc_std},
                    "audio": {"mean": audio_mean, "std":audio_std},
                    "predictions": {"mean": pred_mean, "std":pred_std},

    }
    stats_path = os.path.join(results_path, "statistics.txt")
    statistics_str = str(statistics)
    with open(stats_path, "w") as w:
        w.write(statistics_str)

    # Visualize Emebeddings
    visualize_embeddings(names_to_data['encodings'], outdir=results_path, experiment=args.experiment)

