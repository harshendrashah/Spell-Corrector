from functions import *

testing_dict = {}

test_sen = []

testing_scr = open("testing_script2", "r")

test_sent = testing_scr.read().split(',')
for i in range(0,len(test_sent)-1):
    arr = test_sent[i].split(":")
    str1 = arr[0]
    str2 = arr[1]
    testing_dict[str1] = str2
#print(testing_dict)
print("***************************************************")
# Create your own sentence or use one from the dataset
total = 0
count = 0

for key,value in testing_dict.items():
    text = key
    text = text.lower()
    text = text_to_ints(text)

    exp_output = value
    exp_output = exp_output.lower()
    exp_output = text_to_ints(exp_output)

    #random = np.random.randint(0,len(testing_sorted))
    #text = testing_sorted[random]
    #text = noise_maker(text, 0.95)

    checkpoint = "./kp=0.75,nl=3,th=0.75.ckpt"
    tf.reset_default_graph()
    model, saver = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction) 

    with tf.Session() as sess:
        # Load saved model
        #saver = tf.train.Saver()
        saver.restore(sess, './kp=0.75,nl=3,th=0.75.ckpt')

        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size, 
                                                    model.inputs_length: [len(text)]*batch_size,
                                                    model.targets_length: [len(text)+1], 
                                                    model.keep_prob: [1.0]})[0]

    # Remove the padding from the generated sentence
    pad = vocab_to_int["<PAD>"] 

    print('\nText')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

    pred = ""
    exp = ""

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
    pred = "".join([int_to_vocab[i] for i in answer_logits if i != pad])
    
    print('\nExpected_answer')
    print('  Word Ids:    {}'.format([i for i in exp_output]))
    print('  Exp Output Words: {}'.format("".join([int_to_vocab[i] for i in exp_output])))
    exp = "".join([int_to_vocab[i] for i in exp_output])
    
    
    count = count + 1
    seq=difflib.SequenceMatcher(None, pred,exp)
    d=seq.ratio()*100
    total = total + d


print(total/count)