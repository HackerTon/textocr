from abc import ABC


class ExperimentBase(ABC):
    train_dataloader = None
    test_dataloader = None
    model = None
    optimizer = None
    scheduler = None
    preprocessor = None

    def __getitem__(self, key):
        return {
            "train_dataloader": self.train_dataloader,
            "test_dataloader": self.test_dataloader,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "preprocessor": self.preprocessor,
        }[key]
