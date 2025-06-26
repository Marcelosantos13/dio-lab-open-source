#!/usr/bin/env python3
"""
Script de inferência para conversão de voz usando modelos RVC treinados
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import librosa
import soundfile as sf

from rvc_core import RVCModel, AudioProcessor, load_pretrained_model
from utils import AudioUtils, FeatureExtractor, Visualizer
from config import RVCConfig, VoicePresets

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RVCInference:
    """Classe para inferência RVC"""
    
    def __init__(self, model_path: str, config: Optional[RVCConfig] = None):
        self.config = config or RVCConfig()
        self.device = self.config.DEVICE
        
        # Carregar modelo
        logger.info(f"Carregando modelo: {model_path}")
        self.model = load_pretrained_model(model_path, self.device)
        self.model.eval()
        
        # Inicializar processadores
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH
        )
        
        self.feature_extractor = FeatureExtractor(
            sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH
        )
        
        logger.info("Modelo carregado e pronto para inferência")
    
    def convert_voice(self, 
                     source_audio_path: str,
                     output_path: str,
                     target_speaker_audio: Optional[str] = None,
                     pitch_shift: float = 0.0,
                     formant_shift: float = 0.0,
                     voice_preset: Optional[str] = None) -> str:
        """
        Converte voz usando o modelo RVC
        
        Args:
            source_audio_path: Caminho do áudio fonte
            output_path: Caminho de saída
            target_speaker_audio: Áudio de referência do speaker alvo (opcional)
            pitch_shift: Ajuste de pitch em semitons
            formant_shift: Ajuste de formante
            voice_preset: Preset de voz predefinido
            
        Returns:
            Caminho do arquivo convertido
        """
        try:
            # Aplicar preset se especificado
            if voice_preset:
                preset_config = VoicePresets.get_preset(voice_preset)
                if preset_config:
                    pitch_shift = preset_config.get('pitch_shift', pitch_shift)
                    formant_shift = preset_config.get('formant_shift', formant_shift)
                    logger.info(f"Aplicando preset: {voice_preset}")
            
            # Carregar e pré-processar áudio fonte
            logger.info("Carregando áudio fonte...")
            source_audio, sr = AudioUtils.load_audio(source_audio_path, self.config.SAMPLE_RATE)
            
            # Pré-processamento
            source_audio = self._preprocess_audio(source_audio, sr)
            
            # Carregar áudio de referência do speaker (se fornecido)
            target_speaker_mel = None
            if target_speaker_audio:
                logger.info("Carregando áudio de referência do speaker...")
                target_audio, _ = AudioUtils.load_audio(target_speaker_audio, self.config.SAMPLE_RATE)
                target_audio = self._preprocess_audio(target_audio, sr)
                target_speaker_mel = self._audio_to_mel(target_audio)
            
            # Converter áudio para mel-espectrograma
            source_mel = self._audio_to_mel(source_audio)
            
            # Aplicar transformações de pitch
            if pitch_shift != 0.0:
                source_audio = librosa.effects.pitch_shift(
                    source_audio, sr=sr, n_steps=pitch_shift
                )
                source_mel = self._audio_to_mel(source_audio)
            
            # Executar conversão RVC
            logger.info("Executando conversão RVC...")
            converted_mel = self._rvc_convert(source_mel, target_speaker_mel)
            
            # Converter mel de volta para áudio
            converted_audio = self._mel_to_audio(converted_mel)
            
            # Aplicar ajuste de formante (pós-processamento)
            if formant_shift != 0.0:
                converted_audio = self._apply_formant_shift(converted_audio, formant_shift)
            
            # Pós-processamento
            converted_audio = self._postprocess_audio(converted_audio)
            
            # Salvar resultado
            AudioUtils.save_audio(converted_audio, output_path, sr)
            logger.info(f"Conversão concluída: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na conversão: {e}")
            raise
    
    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Pré-processa áudio"""
        # Normalizar
        if self.config.NORMALIZE_AUDIO:
            audio = AudioUtils.normalize_audio(audio)
        
        # Remover silêncio
        if self.config.REMOVE_SILENCE:
            audio = AudioUtils.remove_silence(audio, sr)
        
        # Reduzir ruído
        if self.config.REDUCE_NOISE:
            audio = AudioUtils.reduce_noise(audio, sr)
        
        # Aplicar pré-ênfase
        audio = AudioUtils.apply_preemphasis(audio, self.config.PREEMPHASIS_COEFF)
        
        return audio
    
    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pós-processa áudio"""
        # Normalizar
        audio = AudioUtils.normalize_audio(audio, target_db=-20.0)
        
        # Aplicar filtro suave para reduzir artefatos
        from scipy import signal
        b, a = signal.butter(5, 0.95, 'low')
        audio = signal.filtfilt(b, a, audio)
        
        # Garantir range válido
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        """Converte áudio para mel-espectrograma"""
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        return torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
    
    def _mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """Converte mel-espectrograma para áudio"""
        mel_np = mel.squeeze(0).cpu().numpy()
        audio = self.audio_processor.mel_to_audio(mel_np)
        return audio
    
    def _rvc_convert(self, source_mel: torch.Tensor, 
                    target_speaker_mel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Executa conversão RVC"""
        with torch.no_grad():
            # Executar modelo
            if target_speaker_mel is not None:
                predictions = self.model(source_mel, target_speaker_mel=target_speaker_mel)
            else:
                predictions = self.model(source_mel)
            
            return predictions['converted_mel']
    
    def _apply_formant_shift(self, audio: np.ndarray, shift: float) -> np.ndarray:
        """Aplica ajuste de formante"""
        try:
            # Implementação simplificada usando mudança de sample rate
            if shift != 0.0:
                # Calcular fator de mudança
                factor = 1.0 + (shift * 0.1)  # Ajuste empírico
                
                # Aplicar mudança temporal seguida de pitch correction
                audio_shifted = librosa.effects.time_stretch(audio, rate=factor)
                
                # Corrigir pitch para manter frequência fundamental
                pitch_correction = -librosa.hz_to_note(factor, unicode=False) if factor != 1.0 else 0
                if pitch_correction != 0:
                    audio_shifted = librosa.effects.pitch_shift(
                        audio_shifted, sr=self.config.SAMPLE_RATE, n_steps=pitch_correction
                    )
                
                return audio_shifted
            
            return audio
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar formant shift: {e}")
            return audio
    
    def batch_convert(self, 
                     input_files: list,
                     output_dir: str,
                     **kwargs) -> list:
        """Converte múltiplos arquivos em lote"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        converted_files = []
        
        for i, input_file in enumerate(input_files):
            try:
                input_path = Path(input_file)
                output_path = output_dir / f"converted_{input_path.stem}.wav"
                
                logger.info(f"Convertendo {i+1}/{len(input_files)}: {input_path.name}")
                
                result = self.convert_voice(
                    str(input_path),
                    str(output_path),
                    **kwargs
                )
                
                converted_files.append(result)
                
            except Exception as e:
                logger.error(f"Erro ao converter {input_file}: {e}")
                continue
        
        logger.info(f"Conversão em lote concluída: {len(converted_files)} arquivos")
        return converted_files
    
    def analyze_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analisa características do áudio"""
        # Carregar áudio
        audio, sr = AudioUtils.load_audio(audio_path, self.config.SAMPLE_RATE)
        
        # Extrair características
        features = self.feature_extractor.extract_spectral_features(audio)
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        f0, voiced_flag = self.feature_extractor.extract_pitch(audio)
        
        # Calcular estatísticas
        analysis = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'rms_energy': np.mean(features['rms']),
            'spectral_centroid_mean': np.mean(features['spectral_centroid']),
            'spectral_bandwidth_mean': np.mean(features['spectral_bandwidth']),
            'f0_mean': np.nanmean(f0[voiced_flag]),
            'f0_std': np.nanstd(f0[voiced_flag]),
            'voiced_ratio': np.sum(voiced_flag) / len(voiced_flag)
        }
        
        # Gerar visualizações se diretório especificado
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Salvar plots
            fig1 = Visualizer.plot_waveform(audio, sr, "Forma de Onda")
            fig1.savefig(output_path / "waveform.png", dpi=150, bbox_inches='tight')
            
            fig2 = Visualizer.plot_mel_spectrogram(mel_spec, sr, self.config.HOP_LENGTH)
            fig2.savefig(output_path / "mel_spectrogram.png", dpi=150, bbox_inches='tight')
            
            fig3 = Visualizer.plot_pitch(f0, voiced_flag, sr, self.config.HOP_LENGTH)
            fig3.savefig(output_path / "pitch.png", dpi=150, bbox_inches='tight')
            
            # Salvar análise
            import json
            with open(output_path / "analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)
        
        return analysis

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Inferência RVC")
    parser.add_argument("--model", type=str, required=True,
                       help="Caminho do modelo RVC")
    parser.add_argument("--input", type=str, required=True,
                       help="Arquivo de áudio de entrada")
    parser.add_argument("--output", type=str, required=True,
                       help="Arquivo de áudio de saída")
    parser.add_argument("--target-speaker", type=str,
                       help="Áudio de referência do speaker alvo")
    parser.add_argument("--pitch-shift", type=float, default=0.0,
                       help="Ajuste de pitch em semitons")
    parser.add_argument("--formant-shift", type=float, default=0.0,
                       help="Ajuste de formante")
    parser.add_argument("--preset", type=str,
                       choices=VoicePresets.list_presets(),
                       help="Preset de voz predefinido")
    parser.add_argument("--batch", action="store_true",
                       help="Modo de conversão em lote")
    parser.add_argument("--analyze", action="store_true",
                       help="Analisar áudio de entrada")
    
    args = parser.parse_args()
    
    # Verificar se modelo existe
    if not os.path.exists(args.model):
        logger.error(f"Modelo não encontrado: {args.model}")
        return
    
    # Criar instância de inferência
    inference = RVCInference(args.model)
    
    if args.analyze:
        # Modo análise
        analysis_dir = Path(args.output).parent / "analysis"
        analysis = inference.analyze_audio(args.input, str(analysis_dir))
        
        print("\n=== ANÁLISE DO ÁUDIO ===")
        for key, value in analysis.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        print(f"\nVisualizações salvas em: {analysis_dir}")
        
    elif args.batch:
        # Modo lote
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            logger.error("Para modo lote, --input deve ser um diretório")
            return
        
        # Encontrar arquivos de áudio
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(list(input_dir.glob(f"*{ext}")))
        
        if not audio_files:
            logger.error("Nenhum arquivo de áudio encontrado")
            return
        
        # Converter em lote
        converted = inference.batch_convert(
            [str(f) for f in audio_files],
            args.output,
            target_speaker_audio=args.target_speaker,
            pitch_shift=args.pitch_shift,
            formant_shift=args.formant_shift,
            voice_preset=args.preset
        )
        
        print(f"\nConversão em lote concluída: {len(converted)} arquivos")
        
    else:
        # Modo conversão única
        if not os.path.exists(args.input):
            logger.error(f"Arquivo de entrada não encontrado: {args.input}")
            return
        
        # Executar conversão
        result = inference.convert_voice(
            args.input,
            args.output,
            target_speaker_audio=args.target_speaker,
            pitch_shift=args.pitch_shift,
            formant_shift=args.formant_shift,
            voice_preset=args.preset
        )
        
        print(f"\nConversão concluída: {result}")
        
        # Mostrar informações do preset se usado
        if args.preset:
            preset_info = VoicePresets.get_preset(args.preset)
            print(f"Preset aplicado: {preset_info.get('description', args.preset)}")

if __name__ == "__main__":
    main()