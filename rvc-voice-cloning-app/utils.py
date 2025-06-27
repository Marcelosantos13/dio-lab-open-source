import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
import tempfile
import logging
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import noisereduce as nr

logger = logging.getLogger(__name__)

class AudioUtils:
    """Utilitários para processamento de áudio"""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
        """Carrega arquivo de áudio com resampling se necessário"""
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio, sr
        except Exception as e:
            logger.error(f"Erro ao carregar áudio {file_path}: {e}")
            raise
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sr: int = 22050):
        """Salva áudio em arquivo"""
        try:
            sf.write(file_path, audio, sr)
            logger.info(f"Áudio salvo em: {file_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar áudio: {e}")
            raise
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normaliza áudio para um nível específico em dB"""
        # Calcular RMS
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Converter target_db para amplitude linear
            target_rms = 10**(target_db/20)
            
            # Aplicar normalização
            audio = audio * (target_rms / rms)
        
        # Garantir que não ultrapasse [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    @staticmethod
    def remove_silence(audio: np.ndarray, sr: int, 
                      top_db: int = 20, frame_length: int = 2048,
                      hop_length: int = 512) -> np.ndarray:
        """Remove silêncio do início e fim do áudio"""
        try:
            audio_trimmed, _ = librosa.effects.trim(
                audio, 
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            return audio_trimmed
        except Exception as e:
            logger.warning(f"Erro ao remover silêncio: {e}")
            return audio
    
    @staticmethod
    def reduce_noise(audio: np.ndarray, sr: int, 
                    stationary: bool = True, prop_decrease: float = 0.8) -> np.ndarray:
        """Reduz ruído do áudio"""
        try:
            audio_denoised = nr.reduce_noise(
                y=audio, 
                sr=sr, 
                stationary=stationary,
                prop_decrease=prop_decrease
            )
            return audio_denoised
        except Exception as e:
            logger.warning(f"Erro na redução de ruído: {e}")
            return audio
    
    @staticmethod
    def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """Aplica filtro de pré-ênfase"""
        return signal.lfilter([1, -coeff], [1], audio)
    
    @staticmethod
    def split_audio_by_silence(audio: np.ndarray, sr: int, 
                              min_silence_len: float = 0.5,
                              silence_thresh: int = -40) -> List[np.ndarray]:
        """Divide áudio em segmentos baseado em silêncio"""
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        
        # Converter para pydub
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        
        # Dividir por silêncio
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=int(min_silence_len * 1000),  # ms
            silence_thresh=silence_thresh,
            keep_silence=100  # manter 100ms de silêncio
        )
        
        # Converter de volta para numpy
        segments = []
        for chunk in chunks:
            chunk_array = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32767.0
            segments.append(chunk_array)
        
        return segments

class FeatureExtractor:
    """Extrator de características de áudio"""
    
    def __init__(self, sr: int = 22050, n_mels: int = 80, 
                 hop_length: int = 256, win_length: int = 1024):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extrai mel-espectrograma"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=0,
            fmax=self.sr//2
        )
        
        # Converter para escala log
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extrai coeficientes MFCC"""
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        return mfccs
    
    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrai pitch (F0) e flag de voz"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        return f0, voiced_flag
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extrai características espectrais diversas"""
        features = {}
        
        # Centroide espectral
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # Largura de banda espectral
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # Rolloff espectral
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # RMS energy
        features['rms'] = librosa.feature.rms(
            y=audio, hop_length=self.hop_length
        )[0]
        
        return features

class Visualizer:
    """Classe para visualização de dados de áudio"""
    
    @staticmethod
    def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform"):
        """Plota forma de onda"""
        plt.figure(figsize=(12, 4))
        time = np.linspace(0, len(audio)/sr, len(audio))
        plt.plot(time, audio)
        plt.title(title)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sr: int, title: str = "Spectrogram"):
        """Plota espectrograma"""
        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_mel_spectrogram(mel_spec: np.ndarray, sr: int, 
                           hop_length: int = 256, title: str = "Mel Spectrogram"):
        """Plota mel-espectrograma"""
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            mel_spec, 
            sr=sr, 
            hop_length=hop_length,
            x_axis='time', 
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_pitch(f0: np.ndarray, voiced_flag: np.ndarray, 
                  sr: int, hop_length: int = 256, title: str = "Pitch"):
        """Plota contorno de pitch"""
        plt.figure(figsize=(12, 4))
        times = librosa.frames_to_time(
            np.arange(len(f0)), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # Plotar apenas frames com voz
        voiced_times = times[voiced_flag]
        voiced_f0 = f0[voiced_flag]
        
        plt.plot(voiced_times, voiced_f0, 'b-', linewidth=2)
        plt.title(title)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_comparison(original: np.ndarray, converted: np.ndarray, 
                       sr: int, title: str = "Comparison"):
        """Plota comparação entre áudios original e convertido"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Áudio original
        time = np.linspace(0, len(original)/sr, len(original))
        axes[0].plot(time, original)
        axes[0].set_title('Original')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Áudio convertido
        time = np.linspace(0, len(converted)/sr, len(converted))
        axes[1].plot(time, converted)
        axes[1].set_title('Convertido')
        axes[1].set_xlabel('Tempo (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig

class DatasetUtils:
    """Utilitários para preparação de datasets"""
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """Valida se o arquivo de áudio é válido"""
        try:
            audio, sr = librosa.load(file_path, duration=1.0)  # Carregar apenas 1s para teste
            
            # Verificações básicas
            if len(audio) == 0:
                return False
            
            if sr < 8000:  # Sample rate muito baixo
                return False
            
            if np.max(np.abs(audio)) < 0.001:  # Áudio muito baixo
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar arquivo {file_path}: {e}")
            return False
    
    @staticmethod
    def prepare_training_data(audio_files: List[str], 
                            output_dir: str,
                            segment_length: float = 10.0,
                            overlap: float = 2.0) -> List[str]:
        """Prepara dados de treinamento dividindo em segmentos"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        processed_files = []
        
        for i, file_path in enumerate(audio_files):
            try:
                # Carregar áudio
                audio, sr = AudioUtils.load_audio(file_path)
                
                # Pré-processar
                audio = AudioUtils.normalize_audio(audio)
                audio = AudioUtils.remove_silence(audio, sr)
                audio = AudioUtils.reduce_noise(audio, sr)
                
                # Dividir em segmentos
                segment_samples = int(segment_length * sr)
                overlap_samples = int(overlap * sr)
                step = segment_samples - overlap_samples
                
                for j, start in enumerate(range(0, len(audio) - segment_samples, step)):
                    end = start + segment_samples
                    segment = audio[start:end]
                    
                    # Salvar segmento
                    output_file = output_path / f"segment_{i:03d}_{j:03d}.wav"
                    AudioUtils.save_audio(segment, str(output_file), sr)
                    processed_files.append(str(output_file))
                
            except Exception as e:
                logger.error(f"Erro ao processar {file_path}: {e}")
                continue
        
        logger.info(f"Processados {len(processed_files)} segmentos de áudio")
        return processed_files
    
    @staticmethod
    def create_speaker_dataset(speaker_files: Dict[str, List[str]], 
                             output_dir: str) -> Dict[str, List[str]]:
        """Cria dataset organizado por speaker"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        speaker_datasets = {}
        
        for speaker_id, files in speaker_files.items():
            speaker_dir = output_path / speaker_id
            speaker_dir.mkdir(exist_ok=True)
            
            processed_files = DatasetUtils.prepare_training_data(
                files, 
                str(speaker_dir)
            )
            
            speaker_datasets[speaker_id] = processed_files
        
        return speaker_datasets

class ModelUtils:
    """Utilitários para modelos"""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Conta o número de parâmetros treináveis"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """Retorna o tamanho do modelo em MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def save_training_config(config: Dict[str, Any], output_path: str):
        """Salva configuração de treinamento"""
        import json
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_training_config(config_path: str) -> Dict[str, Any]:
        """Carrega configuração de treinamento"""
        import json
        with open(config_path, 'r') as f:
            return json.load(f)

def create_temp_audio_file(audio: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    """Cria arquivo temporário de áudio"""
    temp_file = tempfile.mktemp(suffix=suffix)
    AudioUtils.save_audio(audio, temp_file, sr)
    return temp_file

def cleanup_temp_files(file_paths: List[str]):
    """Remove arquivos temporários"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Erro ao remover arquivo temporário {file_path}: {e}")