from automodel.text import TextClassifier
import os

def main():
    data_path = os.path.join(os.getcwd(), 'aclImdb')
    text_clas = TextClassifier()
    text_clas.fit(data_path)
    #print(len(text_data.train_dataloader))
#text_clas.fit('.')

if __name__=="__main__":
    main()
