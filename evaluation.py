import numpy as np
from sklearn.manifold import TSNE
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

#t-SNE
def tsne(x, y=None, num_vis=1000, dim=2):
    outfile = 'tsne-{}d'.format(dim)
    title = 'TSNE'
    if y is not None:
        freqs = y[:, 1]
        freqs_set = np.unique(freqs)
        freq_select = freqs_set[42] # frequency of 311 hz
        mask = freqs == freq_select
        vis_samples = x[mask]
        outfile += '-{}'.format(freq_select)
        title += ' | Freq: {} Hz'.format(freq_select)
    else:
        n_samples = len(x)
        indices = np.arange(n_samples)
        vis_indices = np.random.choice(indices, num_vis, replace=False)
        vis_samples = x[vis_indices]
    x_embedded = TSNE(n_components=dim).fit_transform(vis_samples)

    if dim == 2:
        plt.scatter(x_embedded[:,0], x_embedded[:,1])
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[:,0], x_embedded[:,1], x_embedded[:,2])
    plt.title(title)
    plt.savefig(outfile+'.png')

# MFCC
def mfcc(x, sr=8000, outfile='', labels_str=None):

    title = 'MFCC'
    if labels_str is not None:
        title += ' | '+labels_str
    mfccs = librosa.feature.mfcc(y=x,sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, fmax=sr/2)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    outfile += 'mfcc.png'
    plt.savefig(outfile)


# Spectrogram (FFT)
def spec(x, y=None, sr=8000, type='mel', outfile='', labels_str=None):

    title = 'Mel Spectrogram'
    if labels_str is not None:
        title += ' | '+labels_str

    if type == 'mel':
        spec = librosa.feature.melspectrogram(y=x, sr=sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(spec,
                                                  ref=np.max),
                                            y_axis='mel',
                                            x_axis='time', sr=sr, fmax=sr/2)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()

        outfile += 'spec_{}'.format(type) + '.png'
        print('Saving spectrogram to {}'.format(outfile))
        plt.savefig(outfile)

def waveform(x, y=None, sr=8000, outfile='', labels_str=None):
    title_1 = 'Monophonic'
    title_2 = 'Stereo'
    if labels_str is not None:
        title_1 += ' | '+labels_str
        title_2 += ' | '+labels_str

    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x, sr=sr)
    plt.title(title_1)

    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x, sr=sr)
    plt.title(title_2)

    outfile += 'waveform.png'

    plt.savefig(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='path to data')
    parser.add_argument('type', type=str)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('-y', '--labels', type=str, help='path to labels', default=None)
    parser.add_argument('--cats', nargs='+', help='label categories in order', default=None)
    parser.add_argument('--outdir', type=str, default='')
    args = parser.parse_args()

    if args.labels is not None and 'json' not in args.labels:
        if args.cats is None:
            parser.error('--cats is required')

    file_type = args.data.split('.')[-1]
    if file_type == 'npy':
        x = np.load(args.data)
    elif file_type == 'wav':
        x, sr = librosa.load(args.data, sr=None)
    if args.labels is not None:
        label_file_type = args.labels.split('.')[-1]
        if label_file_type == 'npy':
            y = np.load(args.labels)
        elif label_file_type == 'json':
            with open(args.labels) as f:
                y = json.load(f)
    else:
        y=None

    if args.type == 'tsne':
        tsne(x, y=y, dim=args.dim)
    else:
        if args.outdir is not '' and not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)
        output_file = args.outdir
        labels_string = None
        if len(x.shape) > 1:
            sample = x[17000]
        else:
            sample = x 
        output_file += args.data.split('/')[-1].split('.')[0]
        if y is not None:
            if type(y) is dict:
                val = y['Freq']
                cat = 'freq'
                labels_string = '{}:{}'.format(cat, val)
            else:
                val = y[17000, 1]
                cat = args.cats[1]
            output_file += '_{}-{}'.format(cat, val)

        if args.type == 'mfcc':
            mfcc(sample, outfile=output_file, labels_str=labels_string)
        elif args.type == 'spec':
            spec(sample, outfile=output_file, labels_str=labels_string)
        elif args.type == 'wave':
            waveform(sample, outfile=output_file, labels_str=labels_string)