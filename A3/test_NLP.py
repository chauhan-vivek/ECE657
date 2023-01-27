# import required packages
from keras.models import load_model
from train_NLP import dataset
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. Load your saved model

	# 2. Load your testing data

	# 3. Run prediction on the test data and print the test accuracy

    filepath_pos = r'data\aclImdb\test\pos\*.txt'
    filepath_neg = r'data\aclImdb\test\neg\*.txt'
    reviews, labels = dataset(filepath_pos, filepath_neg)
    print(len(reviews), len(labels))
    
    #maxSentLen = percentile
    maxSentLen = 1175
    
    labels = np.array(labels).reshape((-1,1))
    labels = to_categorical(labels, num_classes=2)
    truncatedData = [''.join(seq[:maxSentLen]) for seq in reviews]
    
    model = load_model("models/20955627_NLP_model.model")
    
    tokenizer = pickle.load(open('data/token.p', "rb"))
    final_data = tokenizer.texts_to_sequences(truncatedData)
    
    final_data = pad_sequences(final_data, maxlen=maxSentLen, padding='post')
    evaluate = model.evaluate(final_data, labels)
    print("Test Accuracy and Loss is: ", str(evaluate[1] * 100), "% and ", evaluate[0], " respectively.", sep = '')