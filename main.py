import multiprocessing
from argparse import ArgumentParser, Namespace

import nlp
import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.loggers import TensorBoardLogger


class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self, args):
        super(IMDBSentimentClassifier, self).__init__()
        self.lr = args.lr
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.pretrain = args.pretrain
        self.model = transformers.BertForSequenceClassification.from_pretrained(
            self.pretrain
        )
        self.seq_length = args.seq_length

        self.split = args.split
        self.debug = True if args.debug else False
        self.save_hyperparameters()

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(self.pretrain)

        def _prepare_ds(split):
            dset = nlp.load_dataset(
                "imdb", split=f"{split}[:{self.batch_size if self.debug else '5%' }]"
            )

            class TokenizedDataset(torch.utils.data.Dataset):
                """ This is a slow data processing method.
                
                huggingface nlp library does not handle pickling data shards
                when using tokenizer.map() function.
                """

                def __init__(self, seq_length):
                    super(TokenizedDataset, self).__init__()
                    self.dset = dset
                    self.max_length = seq_length

                def __len__(self):
                    return len(self.dset)

                def _tokenize(self, x):
                    x["label"] = torch.tensor(x["label"])
                    x["input_ids"] = torch.tensor(
                        tokenizer.encode(
                            x["text"],
                            max_length=self.max_length,
                            pad_to_max_length=True,
                            truncation=True,
                        )
                    )
                    return x

                def __getitem__(self, idx):
                    return self._tokenize(self.dset[idx])

            return TokenizedDataset(self.seq_length)

        self.train_ds, self.test_ds = map(_prepare_ds, ("train", "test"))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        (logits,) = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.loss(logits, batch["label"]).mean()
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.loss(logits, batch["label"])

        # accuracy
        correct = 0
        _, predicted = torch.max(logits, 1)
        correct += predicted.eq(batch["label"]).sum().item()
        acc = torch.tensor(100.0 * (correct / batch["label"].size(0)))
        if self.on_gpu:
            acc = acc.cuda(batch["input_ids"].device.index)

        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([output["loss"] for output in outputs], 0).mean()
        acc = torch.stack([output["acc"] for output in outputs], 0).mean()
        out = {"val_loss": loss, "val_acc": acc}
        return {**out, "log": out}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, drop_last=False, shuffle=False
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--epoch", default=10, type=int)
        parser.add_argument(
            "-j",
            "--workers",
            default=4,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 4)",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            default=8,
            type=int,
            metavar="N",
            help="mini-batch size (default: 64), this is the total "
            "batch size of all GPUs on the current node when "
            "using Data Parallel or Distributed Data Parallel",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.01,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )

        parser.add_argument("--pretrain", type=str, default="bert-base-uncased")
        parser.add_argument("--seq_length", default=512)
        parser.add_argument("--split", default="train")
        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.distributed_backend == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = IMDBSentimentClassifier(args)
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(
        add_help=False
    )  # MUST add_help=False in order to conflict among child parsers
    parent_parser = pl.Trainer.add_argparse_args(
        parent_parser
    )  # extends existing argparse by default 'Trainer' attributes.
    parent_parser.add_argument(
        "-d", "--debug", action="store_true", help="turn on debugging mode"
    )
    parent_parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parent_parser.add_argument(
        "--seed", type=int, default=711, help="seed for initializing training."
    )
    parser = IMDBSentimentClassifier.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler=True, deterministic=True, max_epochs=90, gpus=1)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
