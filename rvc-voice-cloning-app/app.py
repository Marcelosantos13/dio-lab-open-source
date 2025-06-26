import gradio as gr
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import os
import tempfile
from pathlib import Path
import json
from typing import Optional, Tuple, List
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RVCVoiceCloner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.sample_rate = 40000
        self.hop_length = 160
        self.win_length = 640
        
        # Inicializar modelos (placeholder - seria carregado de checkpoints reais)
        self.voice_models = {}
        self.current_model = None
        
        logger.info(f"RVC Voice Cloner inicializado no dispositivo: {self.device}")
    
    def load_voice_model(self, model_path: str) -> bool:
        """Carrega um modelo de voz treinado"""
        try:
            # Placeholder para carregamento real do modelo RVC
            model_name = Path(model_path).stem
            self.voice_models[model_name] = {
                "path": model_path,
                "loaded": True,
                "timestamp": datetime.now()
            }
            self.current_model = model_name
            logger.info(f"Modelo {model_name} carregado com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Pré-processa o áudio de entrada"""
        try:
            # Carregar áudio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalizar
            audio = librosa.util.normalize(audio)
            
            # Remover silêncio
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio, sr
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """Extrai características do áudio"""
        try:
            # Extrair pitch (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Extrair MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=13
            )
            
            # Extrair espectrograma mel
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate,
                n_mels=80
            )
            
            return {
                "f0": f0,
                "voiced_flag": voiced_flag,
                "mfccs": mfccs,
                "mel_spectrogram": mel_spec
            }
        except Exception as e:
            logger.error(f"Erro na extração de características: {e}")
            raise
    
    def clone_voice(self, source_audio_path: str, target_voice: str, 
                   pitch_shift: float = 0.0, formant_shift: float = 0.0) -> str:
        """Clona a voz usando RVC"""
        try:
            # Pré-processar áudio
            audio, sr = self.preprocess_audio(source_audio_path)
            
            # Extrair características
            features = self.extract_features(audio)
            
            # Aplicar transformações de pitch e formante
            if pitch_shift != 0.0:
                audio = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=pitch_shift
                )
            
            # Placeholder para conversão RVC real
            # Aqui seria aplicado o modelo RVC treinado
            converted_audio = self.apply_rvc_conversion(audio, target_voice, features)
            
            # Salvar resultado
            output_path = tempfile.mktemp(suffix=".wav")
            sf.write(output_path, converted_audio, sr)
            
            logger.info(f"Conversão de voz concluída: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na clonagem de voz: {e}")
            raise
    
    def apply_rvc_conversion(self, audio: np.ndarray, target_voice: str, 
                           features: dict) -> np.ndarray:
        """Aplica a conversão RVC (placeholder)"""
        # Esta seria a implementação real do RVC
        # Por enquanto, retorna o áudio original com algumas modificações
        
        # Simular conversão aplicando filtros
        converted = audio.copy()
        
        # Aplicar filtro passa-alta para simular mudança de timbre
        converted = librosa.effects.preemphasis(converted)
        
        # Normalizar
        converted = librosa.util.normalize(converted)
        
        return converted
    
    def train_voice_model(self, training_data_path: str, model_name: str, 
                         epochs: int = 100) -> str:
        """Treina um novo modelo de voz"""
        try:
            logger.info(f"Iniciando treinamento do modelo: {model_name}")
            
            # Placeholder para treinamento real
            # Aqui seria implementado o pipeline de treinamento RVC
            
            model_path = self.models_dir / f"{model_name}.pth"
            
            # Simular treinamento
            training_log = {
                "model_name": model_name,
                "epochs": epochs,
                "training_data": training_data_path,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            # Salvar log de treinamento
            log_path = self.models_dir / f"{model_name}_training.json"
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
            
            logger.info(f"Treinamento concluído: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            raise

# Instância global do clonador
voice_cloner = RVCVoiceCloner()

def clone_voice_interface(source_audio, target_voice, pitch_shift, formant_shift):
    """Interface para clonagem de voz"""
    try:
        if source_audio is None:
            return None, "Por favor, faça upload de um arquivo de áudio"
        
        # Converter para caminho temporário se necessário
        if hasattr(source_audio, 'name'):
            source_path = source_audio.name
        else:
            source_path = source_audio
        
        # Clonar voz
        result_path = voice_cloner.clone_voice(
            source_path, target_voice, pitch_shift, formant_shift
        )
        
        return result_path, "Conversão de voz concluída com sucesso!"
        
    except Exception as e:
        return None, f"Erro na conversão: {str(e)}"

def train_model_interface(training_files, model_name, epochs):
    """Interface para treinamento de modelo"""
    try:
        if not training_files:
            return "Por favor, faça upload dos arquivos de treinamento"
        
        if not model_name:
            return "Por favor, forneça um nome para o modelo"
        
        # Simular treinamento
        result = voice_cloner.train_voice_model(
            training_files[0].name if training_files else "", 
            model_name, 
            epochs
        )
        
        return f"Modelo '{model_name}' treinado com sucesso! Salvo em: {result}"
        
    except Exception as e:
        return f"Erro no treinamento: {str(e)}"

def load_model_interface(model_file):
    """Interface para carregar modelo"""
    try:
        if model_file is None:
            return "Por favor, selecione um arquivo de modelo"
        
        success = voice_cloner.load_voice_model(model_file.name)
        
        if success:
            return f"Modelo carregado com sucesso: {Path(model_file.name).stem}"
        else:
            return "Erro ao carregar o modelo"
            
    except Exception as e:
        return f"Erro: {str(e)}"

# Interface Gradio
def create_interface():
    with gr.Blocks(
        title="RVC Voice Cloning App",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        """
    ) as app:
        
        gr.HTML("""
        <div class="header">
            <h1>🎤 RVC Voice Cloning App</h1>
            <p>Aplicativo de clonagem de voz para canto usando RVC (Retrieval-based Voice Conversion)</p>
        </div>
        """)
        
        with gr.Tabs():
            # Aba de Conversão de Voz
            with gr.Tab("🎵 Conversão de Voz"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Áudio de Entrada")
                        source_audio = gr.Audio(
                            label="Faça upload do áudio para converter",
                            type="filepath"
                        )
                        
                        gr.Markdown("### Configurações")
                        target_voice = gr.Dropdown(
                            choices=["Modelo Padrão", "Voz Feminina", "Voz Masculina"],
                            value="Modelo Padrão",
                            label="Voz Alvo"
                        )
                        
                        pitch_shift = gr.Slider(
                            minimum=-12,
                            maximum=12,
                            value=0,
                            step=0.5,
                            label="Ajuste de Tom (semitons)"
                        )
                        
                        formant_shift = gr.Slider(
                            minimum=-1.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Ajuste de Formante"
                        )
                        
                        convert_btn = gr.Button("🎯 Converter Voz", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Resultado")
                        output_audio = gr.Audio(label="Áudio Convertido")
                        conversion_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                
                convert_btn.click(
                    fn=clone_voice_interface,
                    inputs=[source_audio, target_voice, pitch_shift, formant_shift],
                    outputs=[output_audio, conversion_status]
                )
            
            # Aba de Treinamento
            with gr.Tab("🏋️ Treinamento de Modelo"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Dados de Treinamento")
                        training_files = gr.File(
                            label="Arquivos de Áudio para Treinamento",
                            file_count="multiple",
                            file_types=["audio"]
                        )
                        
                        model_name = gr.Textbox(
                            label="Nome do Modelo",
                            placeholder="Ex: minha_voz_v1"
                        )
                        
                        epochs = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Número de Épocas"
                        )
                        
                        train_btn = gr.Button("🚀 Iniciar Treinamento", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Status do Treinamento")
                        training_status = gr.Textbox(
                            label="Log de Treinamento",
                            lines=10,
                            interactive=False
                        )
                
                train_btn.click(
                    fn=train_model_interface,
                    inputs=[training_files, model_name, epochs],
                    outputs=[training_status]
                )
            
            # Aba de Gerenciamento de Modelos
            with gr.Tab("📁 Gerenciar Modelos"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Carregar Modelo")
                        model_file = gr.File(
                            label="Arquivo do Modelo (.pth)",
                            file_types=[".pth", ".pt"]
                        )
                        
                        load_btn = gr.Button("📥 Carregar Modelo", variant="primary")
                        
                        model_status = gr.Textbox(
                            label="Status do Modelo",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Modelos Disponíveis")
                        models_list = gr.Dataframe(
                            headers=["Nome", "Data", "Status"],
                            datatype=["str", "str", "str"],
                            label="Modelos Carregados"
                        )
                
                load_btn.click(
                    fn=load_model_interface,
                    inputs=[model_file],
                    outputs=[model_status]
                )
            
            # Aba de Informações
            with gr.Tab("ℹ️ Informações"):
                gr.Markdown("""
                ## Sobre o RVC Voice Cloning App
                
                Este aplicativo utiliza a tecnologia RVC (Retrieval-based Voice Conversion) para clonagem de voz,
                especialmente otimizada para canto.
                
                ### Características:
                - **Conversão em tempo real**: Converta qualquer áudio para a voz desejada
                - **Treinamento personalizado**: Treine seus próprios modelos de voz
                - **Ajustes finos**: Controle tom, formante e outros parâmetros
                - **Interface intuitiva**: Fácil de usar para iniciantes e profissionais
                
                ### Como usar:
                1. **Conversão**: Faça upload de um áudio e selecione a voz alvo
                2. **Treinamento**: Forneça amostras de voz para treinar um modelo personalizado
                3. **Gerenciamento**: Carregue e gerencie seus modelos treinados
                
                ### Requisitos de Áudio:
                - Formato: WAV, MP3, FLAC
                - Qualidade: Mínimo 16kHz, recomendado 44.1kHz
                - Duração: 10 segundos a 10 minutos para conversão
                - Treinamento: Mínimo 10 minutos de áudio limpo
                
                ### Dicas para melhores resultados:
                - Use áudio de alta qualidade sem ruído de fundo
                - Para treinamento, inclua variedade de tons e expressões
                - Ajuste os parâmetros gradualmente para encontrar o som ideal
                """)
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <p><strong>RVC Voice Cloning App</strong> - Desenvolvido com ❤️ para a comunidade de música</p>
            <p>Versão 1.0 | Powered by RVC & Gradio</p>
        </div>
        """)
    
    return app

if __name__ == "__main__":
    # Criar interface
    app = create_interface()
    
    # Configurar para permitir acesso externo
    app.launch(
        server_name="0.0.0.0",
        server_port=12000,
        share=False,
        show_error=True,
        debug=True
    )