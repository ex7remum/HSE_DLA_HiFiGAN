import argparse
import collections
import warnings
import itertools
import numpy as np
import torch

import hw_nv.loss as module_loss
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    d_loss_module = config.init_obj(config["d_loss"], module_loss).to(device)
    g_loss_module = config.init_obj(config["g_loss"], module_loss).to(device)
    # metrics = [
    #     config.init_obj(metric_dict, module_metric)
    #     for metric_dict in config["metrics"]
    # ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler

    trainable_params_g = filter(lambda p: p.requires_grad, model.generator.parameters())
    g_optimizer = config.init_obj(config["g_optimizer"], torch.optim, trainable_params_g)
    lr_g_scheduler = config.init_obj(config["lr_g_scheduler"], torch.optim.lr_scheduler, g_optimizer)

    trainable_params_d = filter(lambda p: p.requires_grad, itertools.chain(model.MPD.parameters(),
                                model.MSD.parameters()))
    d_optimizer = config.init_obj(config["d_optimizer"], torch.optim, trainable_params_d)
    lr_d_scheduler = config.init_obj(config["lr_d_scheduler"], torch.optim.lr_scheduler, d_optimizer)

    trainer = Trainer(
        model,
        g_criterion=g_loss_module,
        d_criterion=d_loss_module,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_g_scheduler=lr_g_scheduler,
        lr_d_scheduler=lr_d_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
