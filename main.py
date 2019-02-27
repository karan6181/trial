from automodel.imageclassifier import ImageClassifier
from automodel.data import Data

file_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/minc-2500-tiny.zip'
path = './minc-2500-tiny/train'

if __name__ == '__main__':
    Data.download(file_url)
    auto = ImageClassifier(path)
    auto.fit()