import os
import random
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import cv2
from tqdm import tqdm
from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config.load_config import TrainConfig
from datasets.transform import transform
from datasets.torch_dataset import WhdldDataset
from models.attunet import UNet
from utils import parse_args, setup_logging
from metrics import Metrics
from tools.cfg import py2cfg
from tools.metric import Evaluator

# 忽略警告
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self,
                 model,
                 num_classes,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 resume,
                 output_dir,
                 config_file):  # 添加 config_file 作为参数
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.resume = resume
        self.model_name = ""
        self.save_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.config_file = config_file  # 保存 config_file

        self.init_settings()
        
    # 如果指定了恢复路径，则加载模型
    def init_settings(self):
        if self.resume:
            self.load_model(self.resume)
            self.model_name = os.path.basename(self.resume)
            
        logger.info('init output dirs ... ')
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info('init loggings ... ')
        setup_logging(self.save_dir)
        
        logger.info('init Metrics ... ')
        self.metrics = Metrics(self.num_classes, self.device)



    def train(self, epochs):
        best_val_acc = 0.0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            train_loss = 0.0

            for i, data in enumerate(self.train_loader):
                images = data['image']
                labels = data['label']
                images, labels = images.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                logger.info(f"[{epoch + 1}/{epochs}][{i}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
            
            self.scheduler.step()
            train_loss /= len(self.train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}")

            evaluate_results = self.evaluate()
            val_acc = evaluate_results['total_acc']
            val_miou = evaluate_results['total_iou']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"epoch_{epoch + 1}_acc_{val_acc:.4f}_miou_{val_miou:.4f}.pth")
                
                
    def evaluate(self):
        self.model.eval()
        self.metrics.update()
        val_loss = 0.0
        with torch.no_grad():
            logger.info("start evaluating ...")
            for data in tqdm(self.val_loader):
                images = data['image']
                labels = data['label']
                images, labels = images.to(self.device), labels.to(self.device).long()
                outputs = self.model(images)
                _, preds = torch.max(outputs, dim=1)
                # 更新混淆矩阵
                self.metrics.sample_add(labels, preds)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(self.val_loader.dataset)
        logger.info(f"Validation Loss: {val_loss:.4f}")

        # 计算并记录指标
        results = self.metrics.compute()
        logger.info("Validation Metrics:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")

        return results

    def save_model(self, filename):
        if os.path.exists(os.path.join(self.save_dir, self.model_name)) and self.model_name != "":
            os.remove(os.path.join(self.save_dir, self.model_name))
        model_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        self.model_name = filename
        logger.info(f"Model saved to {model_path}")

    def load_model(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.to(self.device)
        logger.info(f"Model loaded from {checkpoint}")
        
def main(config_file):
    train_config = TrainConfig(config_file)
    config = train_config.config
    
    # 初始化模型、损失函数、优化器
    device = train_config.device
    train_loader = train_config.train_loader
    val_loader = train_config.val_loader
    model = train_config.model
    criterion = train_config.loss_function
    optimizer = train_config.optimizer
    scheduler = train_config.scheduler
    
    # 创建 Trainer 实例
    trainer = Trainer(model,
                      config['dataset']['num_classes'],
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      scheduler,
                      device,
                      config['training']['resume'],
                      config['save']['output_dir'],
                      config_file)  # 传递 config_file 参数
    
    # 开始训练
    trainer.train(config['training']['epochs'])


if __name__ == "__main__":
    config_file = './config/config_model.yaml'  # 替换为你的 YAML 文件路径
    main(config_file)
