import sys
import io

class ExtendedNaiveBayes(object):

    def __init__(self, trainingData):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """
        train = trainingData.splitlines()
        self.red_dict = {}
        self.blue_dict = {}
        self.red_bigram = {}
        self.blue_bigram = {}
        for sentence in train:
            label, content = sentence.split('\t')
            token_list = content.split(' ')
            for token in token_list:
                if label == 'RED':
                    if token not in self.red_dict:
                        self.red_dict[token] = 1
                    else:
                        self.red_dict[token] += 1
                if label == 'BLUE':
                    if token not in self.blue_dict:
                        self.blue_dict[token] = 1
                    else:
                        self.blue_dict[token] += 1
        
        for sentence in train:
            label, content = sentence.split('\t')
            token_list = content.split(' ')
            for idx in range(len(token_list)-1):
                token = token_list[idx] + ' ' + token_list[idx+1]
                if label == 'RED':
                    if token not in self.red_bigram:
                        self.red_bigram[token] = 1
                    else:
                        self.red_bigram[token] += 1
                if label == 'BLUE':
                    if token not in self.blue_bigram:
                        self.blue_bigram[token] = 1
                    else:
                        self.blue_bigram[token] += 1
        
        red_sum = 0
        for v in self.red_dict.values():
            red_sum += v
        blue_sum = 0
        for v in self.blue_dict.values():
            blue_sum += v

        red_distinct = len(self.red_dict)
        blue_distinct = len(self.blue_dict)

        for v in self.red_dict:
            self.red_dict[v] = (self.red_dict[v] + 1) / (red_sum + red_distinct)
        for v in self.blue_dict:
            self.blue_dict[v] = (self.blue_dict[v] + 1) / (blue_sum + blue_distinct)
        
        self.p_red = red_sum / (red_sum + blue_sum)
        self.p_blue = blue_sum / (red_sum + blue_sum)
        self.red_denominator = red_sum + red_distinct
        self.blue_denominator = blue_sum + blue_distinct
        
        red_bi_sum = 0
        for v in self.red_bigram.values():
            red_bi_sum += v
        blue_bi_sum = 0
        for v in self.blue_bigram.values():
            blue_bi_sum += v

        red_bi_distinct = len(self.red_bigram)
        blue_bi_distinct = len(self.blue_bigram)

        for v in self.red_bigram:
            self.red_bigram[v] = (self.red_bigram[v] + 1) / (red_bi_sum + red_bi_distinct)
        for v in self.blue_bigram:
            self.blue_bigram[v] = (self.blue_bigram[v] + 1) / (blue_bi_sum + blue_bi_distinct)
            
        self.p_red_bi = red_bi_sum / (red_bi_sum + blue_bi_sum)
        self.p_blue_bi = blue_bi_sum / (red_bi_sum + blue_bi_sum)
        self.red_denominator_bi = red_bi_sum + red_bi_distinct
        self.blue_denominator_bi = blue_bi_sum + blue_bi_distinct

    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """
        token_list = sentence.split(' ')
        red = math.log(self.p_red_bi)
        blue = math.log(self.p_blue_bi)
        for idx in range(len(token_list)-1):
            token = token_list[idx]
            bi = token_list[idx] + ' ' + token_list[idx+1]
            if bi in self.red_bigram:
                red += math.log(self.red_bigram[bi])
            elif token in self.red_dict:
                red += math.log(0.01 * self.red_dict[token])
            else:
                red += math.log(1 / self.red_denominator)

            if bi in self.blue_bigram:
                blue += math.log(self.blue_bigram[bi])
            elif token in self.blue_dict:
                blue += math.log(0.01 * self.blue_dict[token])
            else:
                blue += math.log(1 / self.blue_denominator)
        return {'red': red, 'blue': blue}

    def testModel(self, testData):
        """
        Using the naive bayes model generated in __init__, test the model using the test data. You should calculate accuracy, precision for each category, and recall for each category. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary.
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """
        test = testData.splitlines()
        red_count = 0
        blue_count = 0
        red_true = 0
        red_false = 0
        blue_true = 0
        blue_false = 0

        for sentence in test:
            red = 0
            blue = 0
            label, content = sentence.split('\t')
            p = self.estimateLogProbability(content)
            if p['red'] > p['blue']:
                red = 1
            else:
                 blue = 1

            if label == 'RED':
                red_count += 1
                if red == 1:
                    red_true += 1
                elif red == 0:
                    red_false += 1
            if label == 'BLUE':
                blue_count += 1
                if blue == 1:
                    blue_true += 1
                elif blue == 0:
                    blue_false += 1
        
        acc = (red_true + blue_true) / (red_count + blue_count)
        pre_red = red_true / (red_true + blue_false)
        rec_red = red_true / (red_true + red_false)
        pre_blue = blue_true / (blue_true + red_false)
        rec_blue = blue_true / (blue_true + blue_false)
            
        return {'overall accuracy': acc,
                'precision for red': pre_red,
                'precision for blue': pre_blue,
                'recall for red': rec_red,
                'recall for blue': rec_blue}

"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 extended.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with io.open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()

    with io.open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = ExtendedNaiveBayes(train_data)
    evaluation = model.testModel(test_data)
    print("overall accuracy: " + str(evaluation['overall accuracy'])
        + "\nprecision for red: " + str(evaluation['precision for red'])
        + "\nprecision for blue: " + str(evaluation['precision for blue'])
        + "\nrecall for red: " + str(evaluation['recall for red'])
        + "\nrecall for blue: " + str(evaluation['recall for blue']))



