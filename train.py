import numpy as np
import tensorflow as tf
from data_util import build_word_dict, build_dataset, batch_iter
from rnn_lm import RNNLanguageModel

def train(train_data, test_data, vocab_size, embedding_size, num_layers, num_hidden,
          learning_rate, keep_prob, batch_size, num_epochs):
    with tf.Session() as sess:
        model = RNNLanguageModel(vocab_size, embedding_size, num_layers, num_hidden)

        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge([loss_summary])
        train_summary_writer = tf.summary.FileWriter("train", sess.graph)
        test_summary_writer = tf.summary.FileWriter("test", sess.graph)

        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x, model.keep_prob: keep_prob}
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss],
                                                feed_dict=feed_dict)
            train_summary_writer.add_summary(summaries, step)

            if step % 100 == 1:
                print("step {0}: loss = {1}".format(step, loss))

        def test_perplexity(test_data, step):
            test_batches = batch_iter(test_data, batch_size, 1)
            losses, iters = 0, 0

            for test_batch_x in test_batches:
                feed_dict = {model.x: test_batch_x, model.keep_prob: keep_prob}
                summaries, loss = sess.run([summary_op, model.loss], feed_dict=feed_dict)
                test_summary_writer.add_summary(summaries, step)
                losses += loss
                iters += 1
            return np.exp(losses / iters)

        batches = batch_iter(train_data, batch_size, num_epochs)
        for batch_x in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_step)

            if step % 100 == 1:
                perplexity = test_perplexity(test_data, step)
                print("\ttest perplexity: {}".format(perplexity))

if __name__ == "__main__":
    embedding_size = 300
    num_layers = 1
    num_hidden = 150
    keep_prob = 0.5
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 30

    train_file = "ptb_data/ptb.train.txt"
    test_file = "ptb_data/ptb.test.txt"
    word_dict = build_word_dict(train_file)
    train_data = build_dataset(train_file, word_dict)
    test_data = build_dataset(test_file, word_dict)

    train(train_data, test_data, len(word_dict), embedding_size, num_layers, num_hidden,
          learning_rate, keep_prob, batch_size, num_epochs)