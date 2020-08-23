import numpy as np
import random

def shuffle_data(x_train_val, y_train_val, x_test, y_test):
    shuffle_indices = np.random.permutation(np.arange(len(y_train_val)))
    shuffled_x = np.asarray(x_train_val[shuffle_indices])
    shuffled_y = y_train_val[shuffle_indices]
    val_sample_index = -1 * int(0.1 * float(len(y_train_val)))  # -5000
    x_train, x_val = shuffled_x[:val_sample_index], shuffled_x[val_sample_index:]
    y_train, y_val = shuffled_y[:val_sample_index], shuffled_y[val_sample_index:]
    x_test = np.asarray(x_test)
    y_train_one_hot = np.eye(10)[y_train]  # [9, 8] -> [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]
    y_train_one_hot = np.squeeze(y_train_one_hot, axis=1)  # (45000, 10)
    y_test_one_hot = np.eye(10)[y_test]
    y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
    y_val_one_hot = np.eye(10)[y_val]
    y_val_one_hot = np.squeeze(y_val_one_hot, axis=1)
    return x_train, y_train_one_hot, x_test, y_test_one_hot, x_val, y_val_one_hot

def batch_iter(x, y , batch_size, num_epochs, shuffle=True):
    num_batches_per_epoch = int((len(x) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(len(x)))
        shuffled_x = x[shuffle_indices]
        shuffled_y = y[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(y))
        yield list(zip(data_augmentation(shuffled_x[start_index:end_index], 4),
                       shuffled_y[start_index:end_index]))

def data_augmentation (x_batch, padding=None):
    for i in range(len(x_batch)):
        if bool(random.getrandbits(1)):
            x_batch[i] = np.fliplr(x_batch[i])
    oshape = np.shape(x_batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(x_batch)):
            new_batch.append(x_batch[i])
            if padding:
                new_batch[i] = np.lib.pad(x_batch[i], pad_width=npad, mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - 32)
            nw = random.randint(0, oshape[1] - 32)
            new_batch[i] = new_batch[i][nh:nh + 32, nw:nw + 32]
        return new_batch