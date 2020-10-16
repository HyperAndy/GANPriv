import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_path", type=str, default="../../RucPriv/BC-TCGA/", help="path of train data")
        self.parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        self.parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
        self.parser.add_argument("--size", type=int, default=17814, help="size of each data dimension")




