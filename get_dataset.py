import os
import tarfile
import wget

if __name__ == "__main__":
    out_dir = './data'
    os.makedirs(out_dir, exist_ok=False)

    print('Downolading dataset')
    filename = wget.download('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                             out=out_dir+'/LJSpeech-1.1.tar.bz2')

    print('Extracting audios')
    with tarfile.open('./data/LJSpeech-1.1.tar.bz2', 'r') as tar:
        tar.extractall(out_dir)
