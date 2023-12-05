import argparse
from re import A
import torch
import numpy as np
from models.vit_lit import VitModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
from models.modeling import  CONFIGS
from utils.data_utils import get_loader

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    
    num_classes = 10 if args.dataset == "cifar10" else 100
    
    config['rank'] = args.rank
    config['input_permutation'] = args.input_permutation

    model = VitModel(config, args=args, num_classes=num_classes)
    model.model.load_from(np.load("attention_data/ViT-B_16-224.npz"))

    # model = VitModel.load_from_checkpoint('runs/cifar10-100_500_rank64/version_2/checkpoints/cifar10-100_500_rank64-epoch=19-train_loss=3.13e-01.ckpt', strict=True, config = config)


    return args, model

def train(args, model):
    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
                                        save_top_k=4, 
                                        monitor='train_loss',
                                        filename=args.name + "-{epoch:02d}-{train_loss:.2e}")
    
    

    args.true_steps = args.epochs * len(train_loader) // len(args.device) // args.gradient_accumulation_steps
    args.warmup_steps = args.warmup_epochs * len(train_loader)// len(args.device) // args.gradient_accumulation_steps
    
    print(f'Steps: {args.true_steps}, Warmup steps:{args.warmup_steps}')

    print(f'Train for {args.epochs} epochs on GPU: {args.device}')
    if len(args.device) == 1:
        trainer = Trainer(accelerator="gpu", 
                    log_every_n_steps=1,
                    devices=1,
                    max_epochs=args.epochs,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=args.name),
                    callbacks=[checkpoint_callback, lr_monitor])
    else:
        trainer = Trainer(accelerator="gpu", 
                    log_every_n_steps=1,
                    devices=args.device,
                    strategy=DDPStrategy(find_unused_parameters=False),
                    max_epochs=args.epochs,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=args.name),
                    callbacks=[checkpoint_callback, lr_monitor])
    
    trainer.validate(model=model, dataloaders=test_loader)

    # trainer.fit(model=model, tsrain_dataloaders=test_loader, val_dataloaders=test_loader)


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    

def main(cmd=''):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16_iff","ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=500, type=int,
                        help="Epochs of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    if cmd != '':
        args = parser.parse_args(cmd)
    else:
        args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] ="1"
    args.device = [0,1,2,3]
    args.rank = 64
    
    
    args.input_permutation = [24, 12, 155, 182, 49, 21, 175, 59, 30, 51, 188, 112, 131, 48, 118, 160, 33, 87, 61, 145, 148, 134, 194, 7, 43, 122, 32, 98, 44, 71, 79, 191, 17, 111, 137, 9, 172, 45, 170, 184, 63, 144, 102, 154, 86, 5, 104, 178, 16, 109, 189, 46, 100, 149, 40, 190, 8, 128, 64, 70, 108, 132, 35, 123, 151, 20, 90, 136, 143, 13, 85, 96, 164, 72, 57, 76, 6, 58, 179, 65, 162, 153, 183, 84, 114, 185, 3, 77, 169, 66, 27, 73, 192, 163, 34, 121, 138, 78, 174, 142, 91, 165, 36, 56, 94, 47, 31, 37, 18, 52, 173, 156, 117, 80, 187, 115, 68, 22, 14, 126, 1, 19, 101, 140, 177, 107, 74, 69, 15, 55, 2, 26, 129, 120, 10, 105, 119, 186, 139, 28, 166, 39, 157, 146, 23, 82, 89, 92, 152, 53, 11, 41, 113, 193, 147, 38, 106, 4, 88, 125, 167, 103, 168, 0, 25, 180, 50, 60, 93, 159, 158, 161, 83, 81, 171, 116, 62, 75, 133, 97, 135, 110, 99, 150, 42, 127, 124, 176, 181, 195, 141, 54, 95, 29, 67, 130]


    # Model & Tokenizer Setup
    args, model = setup(args)
    # return args, model
    # Training

    train(args, model)

    # Full model is 0.9887 accuracy


if __name__ == "__main__":
    main()
