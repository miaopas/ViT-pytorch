import pytorch_lightning as pl
import torch
from torch import nn
from models.modeling import VisionTransformer, CONFIGS
import numpy as np
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VitModel(pl.LightningModule):
    def __init__(self, config, args, num_classes):
        super().__init__() 
        self.args = args
        self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
        self.save_hyperparameters()

    def forward(self, x, labels=None):
        return self.model(x, labels=labels)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                lr=self.args.learning_rate,
                                momentum=0.9,
                                weight_decay=self.args.weight_decay)

        t_total = self.args.true_steps
        if self.args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)
        else:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)

        lr_scheduler = {
                        "scheduler": scheduler, 
                        "interval": "step",
                    }
        return {"optimizer": optimizer, "lr_scheduler":lr_scheduler}

    def training_step(self, batch, batch_idx, loss=nn.MSELoss()):
        x, y = batch
        loss = self(x, y)

        # self.lr_schedulers().step()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # def training_step_end(self, training_step_outputs):
    #     for n, p in self.named_parameters():
    #         if p.grad is None:
    #             print(f'{n} has no grad')


    def validation_step(self, batch, batch_idx, loss=nn.MSELoss()):
        eval_losses = AverageMeter()

        loss_fct = torch.nn.CrossEntropyLoss()
        batch = tuple(t for t in batch)
        x, y = batch
        logits = self(x)[0]

        eval_loss = loss_fct(logits, y)
        eval_losses.update(eval_loss.item())

        preds = torch.argmax(logits, dim=-1)

        self.log("valid_loss", eval_losses.val, prog_bar=True, logger=True, sync_dist=True)
        return preds.detach().cpu().numpy(), y.detach().cpu().numpy()

    def validation_epoch_end(self, outputs):
        pred = [out[0] for out in outputs]
        label = [out[1] for out in outputs]

        all_preds, all_label = np.concatenate(pred), np.concatenate(label)
        accuracy = simple_accuracy(all_preds, all_label)
        
        self.log("valid_accuracy", accuracy, prog_bar=True, logger=True, sync_dist=True)

    def predict(self, x):
        pass
                
