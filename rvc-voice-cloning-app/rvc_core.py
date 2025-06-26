import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Bloco residual para o modelo RVC"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)

class ContentEncoder(nn.Module):
    """Encoder de conteúdo para extrair características linguísticas"""
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, 1)
        
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dilation=2**i) 
            for i in range(num_layers)
        ])
        
        self.output_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
    def forward(self, x):
        x = self.input_conv(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_conv(x)

class SpeakerEncoder(nn.Module):
    """Encoder de speaker para extrair características do falante"""
    def __init__(self, input_dim: int = 80, embedding_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, 128, 5, stride=2, padding=2),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.Conv1d(256, 512, 5, stride=2, padding=2),
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(1, 128),
            nn.GroupNorm(1, 256),
            nn.GroupNorm(1, 512),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = F.gelu(norm(conv(x)))
        
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class Decoder(nn.Module):
    """Decoder para gerar o áudio convertido"""
    def __init__(self, content_dim: int = 256, speaker_dim: int = 256, 
                 output_dim: int = 80, num_layers: int = 8):
        super().__init__()
        
        # Projeção das características do speaker
        self.speaker_proj = nn.Linear(speaker_dim, content_dim)
        
        # Camadas de decodificação
        self.layers = nn.ModuleList([
            ResidualBlock(content_dim, dilation=2**(i%4)) 
            for i in range(num_layers)
        ])
        
        # Camada de saída
        self.output_conv = nn.Conv1d(content_dim, output_dim, 1)
        
    def forward(self, content_features, speaker_embedding):
        # Expandir embedding do speaker para todas as posições temporais
        speaker_features = self.speaker_proj(speaker_embedding).unsqueeze(-1)
        speaker_features = speaker_features.expand(-1, -1, content_features.size(-1))
        
        # Combinar características de conteúdo e speaker
        x = content_features + speaker_features
        
        # Aplicar camadas de decodificação
        for layer in self.layers:
            x = layer(x)
        
        return self.output_conv(x)

class PitchPredictor(nn.Module):
    """Preditor de pitch para controle fino"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, 1, 1),
        ])
        
    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.conv_layers) - 1:
                x = F.gelu(x)
        return x

class RVCModel(nn.Module):
    """Modelo principal RVC para conversão de voz"""
    def __init__(self, 
                 mel_dim: int = 80,
                 content_dim: int = 256,
                 speaker_dim: int = 256,
                 num_speakers: int = 1):
        super().__init__()
        
        self.content_encoder = ContentEncoder(mel_dim, content_dim)
        self.speaker_encoder = SpeakerEncoder(mel_dim, speaker_dim)
        self.decoder = Decoder(content_dim, speaker_dim, mel_dim)
        self.pitch_predictor = PitchPredictor(content_dim)
        
        # Embedding de speakers (para multi-speaker)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_dim)
        else:
            self.speaker_embedding = None
    
    def forward(self, source_mel, target_speaker_id=None, target_speaker_mel=None):
        # Extrair características de conteúdo
        content_features = self.content_encoder(source_mel)
        
        # Obter embedding do speaker alvo
        if target_speaker_id is not None and self.speaker_embedding is not None:
            speaker_embedding = self.speaker_embedding(target_speaker_id)
        elif target_speaker_mel is not None:
            speaker_embedding = self.speaker_encoder(target_speaker_mel)
        else:
            # Usar o próprio áudio como referência
            speaker_embedding = self.speaker_encoder(source_mel)
        
        # Predizer pitch
        predicted_pitch = self.pitch_predictor(content_features)
        
        # Decodificar para mel-espectrograma
        converted_mel = self.decoder(content_features, speaker_embedding)
        
        return {
            'converted_mel': converted_mel,
            'predicted_pitch': predicted_pitch,
            'content_features': content_features,
            'speaker_embedding': speaker_embedding
        }

class RVCLoss(nn.Module):
    """Função de perda para treinamento do RVC"""
    def __init__(self, mel_weight: float = 1.0, pitch_weight: float = 0.1, 
                 content_weight: float = 0.5):
        super().__init__()
        self.mel_weight = mel_weight
        self.pitch_weight = pitch_weight
        self.content_weight = content_weight
        
    def forward(self, predictions, targets):
        losses = {}
        
        # Perda de reconstrução do mel-espectrograma
        mel_loss = F.l1_loss(predictions['converted_mel'], targets['target_mel'])
        losses['mel_loss'] = mel_loss
        
        # Perda de pitch (se disponível)
        if 'target_pitch' in targets:
            pitch_loss = F.mse_loss(predictions['predicted_pitch'], targets['target_pitch'])
            losses['pitch_loss'] = pitch_loss
        else:
            pitch_loss = 0
        
        # Perda de preservação de conteúdo
        if 'source_content' in targets:
            content_loss = F.cosine_embedding_loss(
                predictions['content_features'].mean(dim=-1),
                targets['source_content'].mean(dim=-1),
                torch.ones(predictions['content_features'].size(0)).to(predictions['content_features'].device)
            )
            losses['content_loss'] = content_loss
        else:
            content_loss = 0
        
        # Perda total
        total_loss = (self.mel_weight * mel_loss + 
                     self.pitch_weight * pitch_loss + 
                     self.content_weight * content_loss)
        
        losses['total_loss'] = total_loss
        return losses

class AudioProcessor:
    """Processador de áudio para RVC"""
    def __init__(self, sample_rate: int = 22050, n_mels: int = 80, 
                 hop_length: int = 256, win_length: int = 1024):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        
    def audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Converte áudio para mel-espectrograma"""
        # Calcular STFT
        stft = librosa.stft(audio, hop_length=self.hop_length, 
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        
        # Converter para mel
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate, 
            n_fft=self.win_length, 
            n_mels=self.n_mels
        )
        mel = np.dot(mel_basis, magnitude)
        
        # Aplicar log
        mel = np.log(np.maximum(mel, 1e-5))
        
        return mel
    
    def mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Converte mel-espectrograma para áudio (usando vocoder)"""
        # Placeholder - seria implementado com um vocoder real (HiFi-GAN, etc.)
        # Por enquanto, usa Griffin-Lim
        
        # Converter de log-mel para linear
        mel_linear = np.exp(mel)
        
        # Aproximação inversa (simplificada)
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate, 
            n_fft=self.win_length, 
            n_mels=self.n_mels
        )
        
        # Pseudo-inversão
        magnitude = np.linalg.pinv(mel_basis) @ mel_linear
        
        # Griffin-Lim para reconstruir fase
        audio = librosa.griffinlim(magnitude, hop_length=self.hop_length,
                                  win_length=self.win_length)
        
        return audio
    
    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrai pitch do áudio"""
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Interpolar valores não-voiced
        f0_interp = np.interp(
            np.arange(len(f0)),
            np.where(voiced_flag)[0],
            f0[voiced_flag]
        )
        
        return f0_interp, voiced_flag

class RVCTrainer:
    """Classe para treinamento do modelo RVC"""
    def __init__(self, model: RVCModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.processor = AudioProcessor()
        self.criterion = RVCLoss()
        
    def prepare_batch(self, audio_files: list) -> Dict[str, torch.Tensor]:
        """Prepara um batch de dados para treinamento"""
        batch_mels = []
        batch_pitches = []
        
        for audio_file in audio_files:
            # Carregar áudio
            audio, _ = librosa.load(audio_file, sr=self.processor.sample_rate)
            
            # Converter para mel
            mel = self.processor.audio_to_mel(audio)
            
            # Extrair pitch
            pitch, _ = self.processor.extract_pitch(audio)
            
            batch_mels.append(mel)
            batch_pitches.append(pitch)
        
        # Converter para tensors
        batch_mels = torch.FloatTensor(np.array(batch_mels)).to(self.device)
        batch_pitches = torch.FloatTensor(np.array(batch_pitches)).to(self.device)
        
        return {
            'mels': batch_mels,
            'pitches': batch_pitches
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executa um passo de treinamento"""
        self.model.train()
        
        # Forward pass
        predictions = self.model(batch['mels'])
        
        # Calcular perdas
        targets = {
            'target_mel': batch['mels'],  # Auto-encoding para começar
            'target_pitch': batch['pitches'].unsqueeze(1)
        }
        
        losses = self.criterion(predictions, targets)
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

def load_pretrained_model(model_path: str, device: torch.device) -> RVCModel:
    """Carrega um modelo pré-treinado"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = RVCModel(
        mel_dim=checkpoint.get('mel_dim', 80),
        content_dim=checkpoint.get('content_dim', 256),
        speaker_dim=checkpoint.get('speaker_dim', 256)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def save_model(model: RVCModel, path: str, metadata: Dict[str, Any] = None):
    """Salva o modelo treinado"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'mel_dim': 80,
        'content_dim': 256,
        'speaker_dim': 256,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Modelo salvo em: {path}")