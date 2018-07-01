from functions import *

testing_dict = {}

test_sen = []

testing_scr = open("input.txt", "r")
value=""
test_sent = testing_scr.read().split('.')
mod = test_sent[0].split("=");
test_sent[0] = mod[1]
for i in range(0,len(test_sent)):
    test_sent[i] = test_sent[i].replace("+"," ")
    value = value + test_sent[i]+"."
print("***************************************************")
# Create your own sentence or use one from the dataset
total = 0
count = 0
filestr = ""
for arr in test_sent:
    text = arr
    text = text.lower()
    text = text_to_ints(text)


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
    

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
    pred = "".join([int_to_vocab[i] for i in answer_logits if i != pad])

    arr = pred.split(" ");
    for i in arr:
        filestr = filestr+i+" "
    filestr = filestr[:-1]
    filestr = filestr+"."
    
file = open("mod_input.txt",'w+')
file.write(value)
file.close()

file=open("output.txt",'w+')
file.write(filestr)
file.close()
