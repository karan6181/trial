from automodel.text import TextClassifier

def main():
    data_path = '/Users/shaabhn/Desktop/autoML/aclImdb'
    text_clas = TextClassifier()
    text_clas.fit(data_path)
    #print(len(text_data.train_dataloader))
#text_clas.fit('.')

if __name__=="__main__":
    main()