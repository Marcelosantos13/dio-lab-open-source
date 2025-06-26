import os
from pathlib import Path
from typing import Dict, Any, List
import torch

class RVCConfig:
    """Configurações para o sistema RVC"""
    
    # Configurações de áudio
    SAMPLE_RATE = 22050
    N_MELS = 80
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_FFT = 1024
    
    # Configurações do modelo
    CONTENT_DIM = 256
    SPEAKER_DIM = 256
    NUM_LAYERS_ENCODER = 6
    NUM_LAYERS_DECODER = 8
    
    # Configurações de treinamento
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0
    
    # Configurações de perda
    MEL_WEIGHT = 1.0
    PITCH_WEIGHT = 0.1
    CONTENT_WEIGHT = 0.5
    ADVERSARIAL_WEIGHT = 0.1
    
    # Configurações de dados
    SEGMENT_LENGTH = 10.0  # segundos
    OVERLAP = 2.0  # segundos
    MIN_AUDIO_LENGTH = 5.0  # segundos
    MAX_AUDIO_LENGTH = 30.0  # segundos
    
    # Configurações de pré-processamento
    NORMALIZE_AUDIO = True
    REMOVE_SILENCE = True
    REDUCE_NOISE = True
    PREEMPHASIS_COEFF = 0.97
    
    # Configurações de pitch
    F0_MIN = 80.0  # Hz
    F0_MAX = 800.0  # Hz
    PITCH_SHIFT_RANGE = (-12, 12)  # semitons
    
    # Configurações de diretórios
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Configurações de device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2
    
    # Configurações de checkpoint
    SAVE_EVERY = 1000  # steps
    VALIDATE_EVERY = 500  # steps
    KEEP_CHECKPOINTS = 5
    
    # Configurações de logging
    LOG_LEVEL = "INFO"
    TENSORBOARD_LOG = True
    WANDB_LOG = False
    
    @classmethod
    def create_directories(cls):
        """Cria diretórios necessários"""
        for dir_path in [cls.MODELS_DIR, cls.DATA_DIR, cls.LOGS_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Retorna configurações do modelo"""
        return {
            "mel_dim": cls.N_MELS,
            "content_dim": cls.CONTENT_DIM,
            "speaker_dim": cls.SPEAKER_DIM,
            "num_layers_encoder": cls.NUM_LAYERS_ENCODER,
            "num_layers_decoder": cls.NUM_LAYERS_DECODER,
        }
    
    @classmethod
    def get_audio_config(cls) -> Dict[str, Any]:
        """Retorna configurações de áudio"""
        return {
            "sample_rate": cls.SAMPLE_RATE,
            "n_mels": cls.N_MELS,
            "hop_length": cls.HOP_LENGTH,
            "win_length": cls.WIN_LENGTH,
            "n_fft": cls.N_FFT,
        }
    
    @classmethod
    def get_training_config(cls) -> Dict[str, Any]:
        """Retorna configurações de treinamento"""
        return {
            "batch_size": cls.BATCH_SIZE,
            "learning_rate": cls.LEARNING_RATE,
            "num_epochs": cls.NUM_EPOCHS,
            "warmup_steps": cls.WARMUP_STEPS,
            "gradient_clip": cls.GRADIENT_CLIP,
            "device": str(cls.DEVICE),
        }
    
    @classmethod
    def get_loss_config(cls) -> Dict[str, Any]:
        """Retorna configurações de perda"""
        return {
            "mel_weight": cls.MEL_WEIGHT,
            "pitch_weight": cls.PITCH_WEIGHT,
            "content_weight": cls.CONTENT_WEIGHT,
            "adversarial_weight": cls.ADVERSARIAL_WEIGHT,
        }

class PretrainedModels:
    """Configurações para modelos pré-treinados"""
    
    # URLs para download de modelos (placeholder)
    MODELS = {
        "base_rvc": {
            "url": "https://example.com/base_rvc.pth",
            "description": "Modelo base RVC para conversão geral",
            "size": "150MB",
            "languages": ["pt", "en", "es"]
        },
        "singing_rvc": {
            "url": "https://example.com/singing_rvc.pth", 
            "description": "Modelo especializado para canto",
            "size": "200MB",
            "languages": ["pt", "en"]
        },
        "multilingual_rvc": {
            "url": "https://example.com/multilingual_rvc.pth",
            "description": "Modelo multilíngue",
            "size": "300MB", 
            "languages": ["pt", "en", "es", "fr", "de"]
        }
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Retorna informações sobre um modelo"""
        return cls.MODELS.get(model_name, {})
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Lista modelos disponíveis"""
        return list(cls.MODELS.keys())

class VoicePresets:
    """Presets de voz para conversão rápida"""
    
    PRESETS = {
        "feminina_suave": {
            "pitch_shift": 3.0,
            "formant_shift": 0.2,
            "brightness": 1.1,
            "warmth": 0.9,
            "description": "Voz feminina suave e melodiosa"
        },
        "feminina_potente": {
            "pitch_shift": 2.0,
            "formant_shift": 0.1,
            "brightness": 1.3,
            "warmth": 1.0,
            "description": "Voz feminina potente para canto"
        },
        "masculina_grave": {
            "pitch_shift": -4.0,
            "formant_shift": -0.2,
            "brightness": 0.9,
            "warmth": 1.2,
            "description": "Voz masculina grave e encorpada"
        },
        "masculina_tenor": {
            "pitch_shift": -1.0,
            "formant_shift": 0.0,
            "brightness": 1.0,
            "warmth": 1.0,
            "description": "Voz masculina tenor equilibrada"
        },
        "crianca": {
            "pitch_shift": 8.0,
            "formant_shift": 0.4,
            "brightness": 1.4,
            "warmth": 0.8,
            "description": "Voz infantil clara e brilhante"
        },
        "robotica": {
            "pitch_shift": 0.0,
            "formant_shift": 0.0,
            "brightness": 0.7,
            "warmth": 0.5,
            "description": "Efeito robótico/sintético"
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Retorna configurações de um preset"""
        return cls.PRESETS.get(preset_name, {})
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """Lista presets disponíveis"""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_descriptions(cls) -> Dict[str, str]:
        """Retorna descrições dos presets"""
        return {name: preset["description"] for name, preset in cls.PRESETS.items()}

class AudioFormats:
    """Formatos de áudio suportados"""
    
    INPUT_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]
    OUTPUT_FORMATS = [".wav", ".mp3", ".flac"]
    
    QUALITY_SETTINGS = {
        "baixa": {"bitrate": "128k", "sample_rate": 22050},
        "media": {"bitrate": "192k", "sample_rate": 44100},
        "alta": {"bitrate": "320k", "sample_rate": 44100},
        "lossless": {"format": "flac", "sample_rate": 44100}
    }
    
    @classmethod
    def is_supported_input(cls, file_path: str) -> bool:
        """Verifica se o formato de entrada é suportado"""
        return Path(file_path).suffix.lower() in cls.INPUT_FORMATS
    
    @classmethod
    def is_supported_output(cls, file_path: str) -> bool:
        """Verifica se o formato de saída é suportado"""
        return Path(file_path).suffix.lower() in cls.OUTPUT_FORMATS

class AdvancedConfig:
    """Configurações avançadas para usuários experientes"""
    
    # Configurações de modelo avançadas
    USE_ATTENTION = True
    ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.1
    
    # Configurações de otimização
    OPTIMIZER = "AdamW"
    WEIGHT_DECAY = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    
    # Configurações de scheduler
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = 100
    ETA_MIN = 1e-6
    
    # Configurações de augmentação
    USE_AUGMENTATION = True
    PITCH_AUGMENT_RANGE = (-2, 2)  # semitons
    SPEED_AUGMENT_RANGE = (0.9, 1.1)
    NOISE_AUGMENT_PROB = 0.1
    
    # Configurações de validação
    VALIDATION_SPLIT = 0.1
    CROSS_VALIDATION_FOLDS = 5
    
    # Configurações de inferência
    INFERENCE_BATCH_SIZE = 1
    USE_MIXED_PRECISION = True
    COMPILE_MODEL = False  # PyTorch 2.0+
    
    @classmethod
    def get_optimizer_config(cls) -> Dict[str, Any]:
        """Retorna configurações do otimizador"""
        return {
            "optimizer": cls.OPTIMIZER,
            "weight_decay": cls.WEIGHT_DECAY,
            "betas": (cls.BETA1, cls.BETA2),
            "eps": cls.EPS,
        }
    
    @classmethod
    def get_scheduler_config(cls) -> Dict[str, Any]:
        """Retorna configurações do scheduler"""
        return {
            "scheduler": cls.SCHEDULER,
            "T_max": cls.T_MAX,
            "eta_min": cls.ETA_MIN,
        }

# Configuração global
config = RVCConfig()
config.create_directories()

# Exportar configurações principais
__all__ = [
    "RVCConfig",
    "PretrainedModels", 
    "VoicePresets",
    "AudioFormats",
    "AdvancedConfig",
    "config"
]