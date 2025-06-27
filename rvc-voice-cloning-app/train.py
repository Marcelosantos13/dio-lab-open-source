#!/usr/bin/env python3
"""
Script de treinamento para modelos RVC
"""

import argparse
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm import tqdm

from rvc_core import RVCModel, RVCLoss, AudioProcessor, save_model
from utils import AudioUtils, FeatureExtractor, DatasetUtils
from config import RVCConfig, AdvancedConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RVCDataset(Dataset):
    """Dataset para treinamento RVC"""
    
    def __init__(self, audio_files: List[str], config: RVCConfig):
        self.audio_files = audio_files
        self.config = config
        self.processor = AudioProcessor(
            sample_rate=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH
        )
        self.feature_extractor = FeatureExtractor(
            sr=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH
        )
        
        # Validar arquivos
        self.valid_files = []
        for file_path in audio_files:
            if DatasetUtils.validate_audio_file(file_path):
                self.valid_files.append(file_path)
            else:
                logger.warning(f"Arquivo inválido ignorado: {file_path}")
        
        logger.info(f"Dataset criado com {len(self.valid_files)} arquivos válidos")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        
        try:
            # Carregar áudio
            audio, sr = AudioUtils.load_audio(file_path, self.config.SAMPLE_RATE)
            
            # Pré-processar
            if self.config.NORMALIZE_AUDIO:
                audio = AudioUtils.normalize_audio(audio)
            
            if self.config.REMOVE_SILENCE:
                audio = AudioUtils.remove_silence(audio, sr)
            
            if self.config.REDUCE_NOISE:
                audio = AudioUtils.reduce_noise(audio, sr)
            
            # Aplicar augmentação (se habilitada)
            if AdvancedConfig.USE_AUGMENTATION and np.random.random() < 0.5:
                audio = self._apply_augmentation(audio, sr)
            
            # Extrair mel-espectrograma
            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
            
            # Extrair pitch
            f0, voiced_flag = self.feature_extractor.extract_pitch(audio)
            
            # Converter para tensors
            mel_tensor = torch.FloatTensor(mel_spec)
            f0_tensor = torch.FloatTensor(f0)
            voiced_tensor = torch.BoolTensor(voiced_flag)
            
            return {
                'mel': mel_tensor,
                'f0': f0_tensor,
                'voiced': voiced_tensor,
                'file_path': file_path
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar {file_path}: {e}")
            # Retornar item vazio em caso de erro
            return {
                'mel': torch.zeros(self.config.N_MELS, 100),
                'f0': torch.zeros(100),
                'voiced': torch.zeros(100, dtype=torch.bool),
                'file_path': file_path
            }
    
    def _apply_augmentation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Aplica augmentação de dados"""
        # Augmentação de pitch
        if np.random.random() < 0.3:
            pitch_shift = np.random.uniform(*AdvancedConfig.PITCH_AUGMENT_RANGE)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        
        # Augmentação de velocidade
        if np.random.random() < 0.3:
            speed_factor = np.random.uniform(*AdvancedConfig.SPEED_AUGMENT_RANGE)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        # Augmentação de ruído
        if np.random.random() < AdvancedConfig.NOISE_AUGMENT_PROB:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise
        
        return audio

def collate_fn(batch):
    """Função de collate para o DataLoader"""
    # Filtrar itens inválidos
    valid_batch = [item for item in batch if item['mel'].numel() > 0]
    
    if not valid_batch:
        return None
    
    # Encontrar dimensões máximas
    max_time = max(item['mel'].shape[1] for item in valid_batch)
    
    # Pad sequências
    mels = []
    f0s = []
    voiced_flags = []
    
    for item in valid_batch:
        mel = item['mel']
        f0 = item['f0']
        voiced = item['voiced']
        
        # Pad temporal
        if mel.shape[1] < max_time:
            pad_size = max_time - mel.shape[1]
            mel = torch.nn.functional.pad(mel, (0, pad_size))
            f0 = torch.nn.functional.pad(f0, (0, pad_size))
            voiced = torch.nn.functional.pad(voiced, (0, pad_size))
        
        mels.append(mel)
        f0s.append(f0)
        voiced_flags.append(voiced)
    
    return {
        'mels': torch.stack(mels),
        'f0s': torch.stack(f0s),
        'voiced_flags': torch.stack(voiced_flags)
    }

class RVCTrainer:
    """Classe principal para treinamento RVC"""
    
    def __init__(self, config: RVCConfig, model_name: str, output_dir: str):
        self.config = config
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configurar device
        self.device = config.DEVICE
        logger.info(f"Usando device: {self.device}")
        
        # Inicializar modelo
        self.model = RVCModel(**config.get_model_config()).to(self.device)
        
        # Configurar otimizador
        self.optimizer = self._create_optimizer()
        
        # Configurar scheduler
        self.scheduler = self._create_scheduler()
        
        # Configurar critério de perda
        self.criterion = RVCLoss(**config.get_loss_config())
        
        # Configurar logging
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Métricas de treinamento
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"Modelo inicializado com {self._count_parameters()} parâmetros")
    
    def _create_optimizer(self):
        """Cria otimizador"""
        if AdvancedConfig.OPTIMIZER == "AdamW":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                **AdvancedConfig.get_optimizer_config()
            )
        elif AdvancedConfig.OPTIMIZER == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                betas=(AdvancedConfig.BETA1, AdvancedConfig.BETA2),
                eps=AdvancedConfig.EPS
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=0.9
            )
    
    def _create_scheduler(self):
        """Cria scheduler de learning rate"""
        if AdvancedConfig.SCHEDULER == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=AdvancedConfig.T_MAX,
                eta_min=AdvancedConfig.ETA_MIN
            )
        elif AdvancedConfig.SCHEDULER == "StepLR":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
    
    def _count_parameters(self):
        """Conta parâmetros treináveis"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self, train_files: List[str], val_files: Optional[List[str]] = None):
        """Executa treinamento completo"""
        logger.info("Iniciando treinamento...")
        
        # Criar datasets
        train_dataset = RVCDataset(train_files, self.config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = None
        if val_files:
            val_dataset = RVCDataset(val_files, self.config)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                collate_fn=collate_fn
            )
        
        # Loop de treinamento
        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f"Época {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Treinamento
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validação
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader, epoch)
            
            # Atualizar scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss if val_loss else train_loss)
            else:
                self.scheduler.step()
            
            # Salvar checkpoint
            if (epoch + 1) % 10 == 0 or val_loss and val_loss < self.best_loss:
                self._save_checkpoint(epoch, train_loss, val_loss)
                if val_loss and val_loss < self.best_loss:
                    self.best_loss = val_loss
            
            # Log métricas
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        logger.info("Treinamento concluído!")
        self._save_final_model()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Executa uma época de treinamento"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Treinamento Época {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Mover para device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch['mels'])
            
            # Calcular perda
            targets = {
                'target_mel': batch['mels'],
                'target_pitch': batch['f0s'].unsqueeze(1)
            }
            
            losses = self.criterion(predictions, targets)
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.GRADIENT_CLIP
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Atualizar métricas
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log detalhado
            if self.global_step % 100 == 0:
                for loss_name, loss_value in losses.items():
                    if torch.is_tensor(loss_value):
                        self.writer.add_scalar(
                            f'Train/{loss_name}', 
                            loss_value.item(), 
                            self.global_step
                        )
            
            # Atualizar progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        
        return avg_loss
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Executa uma época de validação"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validação Época {epoch+1}")
            
            for batch in pbar:
                if batch is None:
                    continue
                
                # Mover para device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['mels'])
                
                # Calcular perda
                targets = {
                    'target_mel': batch['mels'],
                    'target_pitch': batch['f0s'].unsqueeze(1)
                }
                
                losses = self.criterion(predictions, targets)
                loss = losses['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Validation/Epoch_Loss', avg_loss, epoch)
        
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'global_step': self.global_step,
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint salvo: {checkpoint_path}")
    
    def _save_final_model(self):
        """Salva modelo final"""
        model_path = self.output_dir / f"{self.model_name}.pth"
        
        metadata = {
            'model_name': self.model_name,
            'training_config': self.config.__dict__,
            'total_parameters': self._count_parameters(),
            'best_loss': self.best_loss
        }
        
        save_model(self.model, str(model_path), metadata)
        logger.info(f"Modelo final salvo: {model_path}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Treinamento RVC")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Diretório com dados de treinamento")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Nome do modelo")
    parser.add_argument("--output-dir", type=str, default="./models",
                       help="Diretório de saída")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Tamanho do batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Taxa de aprendizado")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Proporção para validação")
    
    args = parser.parse_args()
    
    # Configurar
    config = RVCConfig()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    # Encontrar arquivos de áudio
    data_dir = Path(args.data_dir)
    audio_files = []
    
    for ext in ['.wav', '.mp3', '.flac']:
        audio_files.extend(list(data_dir.glob(f"**/*{ext}")))
    
    audio_files = [str(f) for f in audio_files]
    
    if not audio_files:
        logger.error(f"Nenhum arquivo de áudio encontrado em {data_dir}")
        return
    
    logger.info(f"Encontrados {len(audio_files)} arquivos de áudio")
    
    # Dividir em treino e validação
    if args.validation_split > 0:
        split_idx = int(len(audio_files) * (1 - args.validation_split))
        train_files = audio_files[:split_idx]
        val_files = audio_files[split_idx:]
    else:
        train_files = audio_files
        val_files = None
    
    # Criar trainer
    trainer = RVCTrainer(config, args.model_name, args.output_dir)
    
    # Executar treinamento
    trainer.train(train_files, val_files)

if __name__ == "__main__":
    main()