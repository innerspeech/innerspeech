import os
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from edit_distance import SequenceMatcher
from dataset import getDatasetLoaders
from model import GRUDecoder, TDNNDecoder, TransformerDecoder

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb
import sentencepiece as spm
from pyctcdecode import build_ctcdecoder
import neural_decoder.lmDecoderUtils as lmDecoderUtils

from dataset import PHONE_DEF_SIL, CHAR_DEF_SIL

import jiwer

# define the LightningModule
class Lightning_GRUDecoder(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.args = args
        self.save_hyperparameters()
        # TODO: move to conf
        self.blank_penalty = np.log(7)
        if self.args["model"]["tokenizer"] == "bpe":
            self.tokenizer = spm.SentencePieceProcessor(model_file=self.args["model"]["tokenizer_path"])
            self.vocabs = [''] + [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        elif self.args["model"]["tokenizer"] == "phoneme":
            self.vocabs = [''] + PHONE_DEF_SIL
        elif self.args["model"]["tokenizer"] == "char":
            self.vocabs = [''] + CHAR_DEF_SIL

        if self.args["model"]["tokenizer"] == "phoneme":
            self.decoder = build_ctcdecoder(
                self.vocabs,
            )
            self.ngramDecoder = lmDecoderUtils.build_lm_decoder(
                self.args["model"]["lm_path"],
                acoustic_scale=0.5, 
                nbest=1, 
                beam=1
            )
        else:
            self.decoder = build_ctcdecoder(
                self.vocabs,
                kenlm_model_path="train.arpa",  # either .arpa or .bin file
            )

    def training_step(self, batch, batch_idx):
        X, y, X_len, y_len, dayIdx, _ = batch

        # OPTIONAL: add white noise and constant offset
        if self.args["model"]["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device="cuda") * self.args["model"]["whiteNoiseSD"]
        if self.args["model"]["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device="cuda")
                * self.args["model"]["constantOffsetSD"]
            )

        # Compute prediction error
        pred = self.model.forward(X, dayIdx)

        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(torch.int32),
            y_len,
        )

        loss = torch.sum(loss)
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.current_epoch % 5 == 0:
            X, y, X_len, y_len, dayIdx, trans = batch

            # Compute prediction error
            pred = self.model.forward(X, dayIdx)

            loss = self.loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(torch.int32),
                y_len,
            )

            loss = torch.sum(loss)
            self.log("dev_loss", loss, sync_dist=True)

            adjustedLens = ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(
                torch.int32
            )

            trueSeq_decodedSeq_text = []

            for iterIdx in range(pred.shape[0]):
                logits = torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :])
                decodedSeq_arpa_text = self.decoder.decode(logits.cpu().detach().numpy())

                decodedSeq = torch.argmax(logits, dim=-1)
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())
                # Get text from LM decoder
                logits = pred[iterIdx].cpu().detach().numpy()
                logits = np.concatenate(
                        [logits[:, 1:], logits[:, 0:1]], axis=-1
                    ) 
                logits = lmDecoderUtils.rearrange_speech_logits(logits[None, :, :], has_sil=True)
                if self.current_epoch == 0:
                    top_nbest = ""
                else:
                    top_nbest = lmDecoderUtils.lm_decode(
                        self.ngramDecoder,
                        logits[0],
                        blankPenalty=self.blank_penalty,
                        returnNBest=False,
                        rescore=False,
                    )

                # Get only the nbests
                # n_best_texts = [nbest[0] for nbest in nbest_outputs]
                # top_nbest = n_best_texts[0]

                if self.args["model"]["tokenizer"] == "bpe":
                    trueSeq_text = self.tokenizer.decode((trueSeq - 1).tolist())
                    decodedSeq_text = self.tokenizer.decode((decodedSeq - 1).tolist())
                elif self.args["model"]["tokenizer"] == "phoneme":
                    trueSeq_text = " ".join([self.vocabs[i] for i in trueSeq])
                    decodedSeq_text = " ".join([self.vocabs[i] for i in decodedSeq])

                trueSeq_decodedSeq_text.append(
                    [
                        self.current_epoch,
                        iterIdx,
                        jiwer.wer(trueSeq_text, decodedSeq_text),
                        jiwer.wer(trans[iterIdx], top_nbest),
                        trueSeq_text,
                        decodedSeq_text,
                        trans[iterIdx],
                        top_nbest,
                    ]
                )
            # get the list of all trueSeq_text
            all_trueSeq_text = [x[-4] for x in trueSeq_decodedSeq_text]
            all_decodedSeq_text = [x[-3] for x in trueSeq_decodedSeq_text]
            all_decodedSeq_arpa_text = [x[-1] for x in trueSeq_decodedSeq_text]

            # calculate the CER and WER with jiwer
            dev_wer = jiwer.wer(all_trueSeq_text, all_decodedSeq_text)
            dev_wer_lm = jiwer.wer(trans, all_decodedSeq_arpa_text)

            self.logger.log_text(
                key="samples",
                columns=["EPOCH", "IDX", "WER", "WER_lm", "trueSeq", "decodedSeq", "trueSeq_lm", "decodedSeq_lm"],
                data=trueSeq_decodedSeq_text,
            )

            self.log("dev_wer", dev_wer, sync_dist=True)
            self.log("dev_wer_lm", dev_wer_lm, sync_dist=True)
            # print(f"epoch: {self.current_epoch} | dev_wer: {dev_wer} | dev_wer_lm: {dev_wer_lm} | dev_loss: {loss}")

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args["model"]["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=self.args["model"]["l2_decay"],
        )

        # return optimizer
    
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.args["model"]["lrEnd"] / self.args["model"]["lrStart"],
            total_iters=self.args["trainer"]["max_epochs"],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step() 


# Parse config with hydra
@hydra.main(version_base=None, config_path="configs", config_name="config_phoneme.yaml")
def main(cfg: DictConfig) -> None:

    # ------------------------------
    # Congif and local logging setup
    # ------------------------------

    # Logging
    # A logger for this file
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    # Log all the configs
    logging.info(OmegaConf.to_yaml(cfg))

    # Add unique config parameters for experiment tracking
    cfg.base.date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg.base.exp_name = f"{cfg.base.date}_{cfg.base.model_name}_{cfg.model.model_type}_{cfg.base.note}"
    cfg.base.exp_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Create the output directory if it does not exist
    os.makedirs(cfg.base.exp_dir, exist_ok=True)

    # Save modified cfg to yaml file
    OmegaConf.save(config=cfg, f=Path(cfg.base.exp_dir) / cfg.base.config_file)

    # Resolve the config for wandb logging
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    # ------------------------------
    # Start the experiment
    # ------------------------------

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.trainer.cuda_visible_devices)
        
    # Seed everything
    seed_everything(cfg.trainer.seed)

    # check if data set has been preprocessed 
    if not os.path.exists(cfg.dataset.preprocessed_path):
        from dataset import preprocess_neural_data
        preprocess_neural_data(data_dir=cfg.dataset.raw_path, 
                           output_dir=cfg.dataset.preprocessed_path,
                           cfg=cfg)

    # Load dataloaders
    trainLoader, devLoader, testLoader, loadedData = getDatasetLoaders(
        cfg.dataset.preprocessed_path,
        cfg.trainer.batch_size,
    )
    if cfg.model.model_type == 'gru':
        decoder_model = GRUDecoder(
            neural_dim=cfg.model.nInputFeatures,
            n_classes=cfg.model.nClasses,
            hidden_dim=cfg.model.nUnits,
            layer_dim=cfg.model.nLayers,
            nDays=len(loadedData["train"]),
            dropout=cfg.model.dropout,
            device="cuda",
            strideLen=cfg.model.strideLen,
            kernelLen=cfg.model.kernelLen,
            gaussianSmoothWidth=cfg.model.gaussianSmoothWidth,
            bidirectional=cfg.model.bidirectional,
        )
    elif cfg.model.model_type == 'tdnn':
        decoder_model = TDNNDecoder(
            neural_dim=cfg.model.nInputFeatures,
            n_classes=cfg.model.nClasses,
            hidden_dim=cfg.model.nUnits,
            layer_dim=cfg.model.nLayers,
            nDays=len(loadedData["train"]),
            dropout=cfg.model.dropout,
            device="cuda",
            strideLen=cfg.model.strideLen,
            kernelLen=cfg.model.kernelLen,
            gaussianSmoothWidth=cfg.model.gaussianSmoothWidth,
            bidirectional=cfg.model.bidirectional,
        )
    elif cfg.model.model_type == 'transformer':
        decoder_model = TransformerDecoder(
            neural_dim=cfg.model.nInputFeatures,
            n_classes=cfg.model.nClasses,
            hidden_dim=cfg.model.nUnits,
            layer_dim=cfg.model.gru_nLayers,
            nDays=len(loadedData["train"]),
            dropout=cfg.model.dropout,
            device="cuda",
            strideLen=cfg.model.strideLen,
            kernelLen=cfg.model.kernelLen,
            gaussianSmoothWidth=cfg.model.gaussianSmoothWidth,
            bidirectional=cfg.model.bidirectional,
            kwargs=cfg.model,
        )

    model = Lightning_GRUDecoder(decoder_model, wandb.config)

    # Configure environment variables for DDP
    if cfg.cluster.nodes > 1:
        os.environ["MASTER_PORT"] = cfg.cluster.master_port
        os.environ["MASTER_ADDR"] = cfg.cluster.master_addr
        os.environ["WORLD_SIZE"] = cfg.cluster.world_size
        os.environ["NODE_RANK"] = cfg.cluster.node_rank

    # Configure the ML logger
    if cfg.trainer.ml_logging == "mlflow":
        import mlflow
        from lightning.pytorch.loggers import MLFlowLogger
        #TODO: test mlflow locally
        mlflow.pytorch.autolog()
        ml_logger = MLFlowLogger(
            experiment_name=cfg.base.exp_name,
            tracking_uri="file:./ml-runs",
        )
    elif cfg.trainer.ml_logging == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        ml_logger = WandbLogger(
            log_model=False,
            name=cfg.base.exp_name,
            project=cfg.trainer.project_name,
            save_dir=cfg.base.exp_dir,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="dev_wer",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
            dirpath=cfg.base.exp_dir,
        )

        lr_monitor = LearningRateMonitor()

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.gpus,
        accelerator="auto",
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.cluster.nodes,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=ml_logger,
        fast_dev_run=cfg.trainer.fast_dev_run,
    )
    # Start training
    trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=devLoader)

if __name__ == "__main__":
    # USAGE: export CUDA_VISIBLE_DEVICES=1; python train_bpe_pl.py --config-name=config_bpe_64.yaml base.note=testing
    main()