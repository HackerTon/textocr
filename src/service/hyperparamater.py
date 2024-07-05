class Hyperparameter:
    def __init__(
        self,
        epoch,
        learning_rate,
        batch_size_test,
        batch_size_train,
        data_path,
    ):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
        self.data_path = data_path
