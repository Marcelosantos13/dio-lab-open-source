# 🎤 RVC Voice Cloning App

Um aplicativo completo de clonagem de voz para canto usando RVC (Retrieval-based Voice Conversion) com interface web moderna e intuitiva.

## 🌟 Características

- **Conversão de Voz em Tempo Real**: Converta qualquer áudio para a voz desejada
- **Treinamento Personalizado**: Treine seus próprios modelos de voz
- **Interface Web Moderna**: Interface intuitiva construída com Gradio
- **Otimizado para Canto**: Especialmente desenvolvido para aplicações musicais
- **Presets de Voz**: Configurações pré-definidas para diferentes tipos de voz
- **Processamento Avançado**: Redução de ruído, normalização e otimizações automáticas

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- CUDA (opcional, para aceleração GPU)
- FFmpeg

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rvc-voice-cloning-app.git
cd rvc-voice-cloning-app

# Instale as dependências
pip install -r requirements.txt

# Execute o aplicativo
python app.py
```

### Instalação com Conda

```bash
# Crie um ambiente conda
conda create -n rvc-app python=3.9
conda activate rvc-app

# Instale PyTorch com CUDA (opcional)
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Instale outras dependências
pip install -r requirements.txt
```

## 🎯 Como Usar

### 1. Conversão de Voz

1. Acesse a aba **"Conversão de Voz"**
2. Faça upload do arquivo de áudio que deseja converter
3. Selecione a voz alvo ou carregue um modelo personalizado
4. Ajuste os parâmetros (tom, formante) conforme necessário
5. Clique em **"Converter Voz"**
6. Baixe o resultado convertido

### 2. Treinamento de Modelo

1. Acesse a aba **"Treinamento de Modelo"**
2. Faça upload dos arquivos de áudio para treinamento (mínimo 10 minutos)
3. Defina um nome para o modelo
4. Configure o número de épocas de treinamento
5. Clique em **"Iniciar Treinamento"**
6. Acompanhe o progresso no log de treinamento

### 3. Gerenciamento de Modelos

1. Acesse a aba **"Gerenciar Modelos"**
2. Carregue modelos pré-treinados (.pth)
3. Visualize modelos disponíveis
4. Gerencie seus modelos personalizados

## 📁 Estrutura do Projeto

```
rvc-voice-cloning-app/
├── app.py                 # Aplicativo principal Gradio
├── rvc_core.py           # Implementação do modelo RVC
├── utils.py              # Utilitários de processamento
├── config.py             # Configurações do sistema
├── requirements.txt      # Dependências Python
├── models/               # Modelos treinados
├── data/                 # Dados de treinamento
├── logs/                 # Logs de treinamento
├── temp/                 # Arquivos temporários
└── README.md            # Este arquivo
```

## ⚙️ Configuração

### Configurações Básicas

Edite o arquivo `config.py` para personalizar:

- **Sample Rate**: Taxa de amostragem (padrão: 22050 Hz)
- **Batch Size**: Tamanho do batch para treinamento
- **Learning Rate**: Taxa de aprendizado
- **Device**: CPU ou CUDA

### Presets de Voz

O aplicativo inclui presets pré-configurados:

- **Feminina Suave**: Voz feminina melodiosa
- **Feminina Potente**: Voz feminina para canto
- **Masculina Grave**: Voz masculina encorpada
- **Masculina Tenor**: Voz masculina equilibrada
- **Criança**: Voz infantil clara
- **Robótica**: Efeito sintético

## 🎵 Dicas para Melhores Resultados

### Para Conversão

- Use áudio de alta qualidade (mínimo 16kHz, recomendado 44.1kHz)
- Evite ruído de fundo
- Duração ideal: 10 segundos a 10 minutos
- Ajuste gradualmente os parâmetros

### Para Treinamento

- Forneça pelo menos 10 minutos de áudio limpo
- Inclua variedade de tons e expressões
- Use áudio sem reverb ou efeitos
- Mantenha qualidade consistente

## 🔧 Parâmetros de Conversão

### Ajuste de Tom (Pitch Shift)
- **Range**: -12 a +12 semitons
- **Uso**: Alterar a altura da voz
- **Dica**: +3 para feminizar, -4 para masculinizar

### Ajuste de Formante
- **Range**: -1.0 a +1.0
- **Uso**: Alterar o timbre da voz
- **Dica**: Valores positivos para voz mais brilhante

## 📊 Monitoramento

### Logs de Treinamento

O aplicativo gera logs detalhados incluindo:

- Perda de reconstrução
- Perda de pitch
- Perda de conteúdo
- Métricas de validação

### Visualizações

- Forma de onda
- Espectrograma
- Mel-espectrograma
- Contorno de pitch

## 🚨 Solução de Problemas

### Problemas Comuns

**Erro de CUDA**
```bash
# Instale PyTorch com CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Erro de FFmpeg**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Baixe de https://ffmpeg.org/download.html
```

**Memória Insuficiente**
- Reduza o batch_size em config.py
- Use áudio mais curto para treinamento
- Feche outros aplicativos

### Logs de Debug

Para debug detalhado, execute:

```bash
python app.py --debug --log-level DEBUG
```

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **RVC Community**: Pela tecnologia base
- **Gradio Team**: Pela excelente framework de UI
- **PyTorch Team**: Pelo framework de deep learning
- **Librosa**: Pela biblioteca de processamento de áudio

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/seu-usuario/rvc-voice-cloning-app/issues)
- **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/rvc-voice-cloning-app/discussions)
- **Email**: suporte@rvc-app.com

## 🔮 Roadmap

### Versão 1.1
- [ ] Suporte a múltiplos speakers
- [ ] Conversão em lote
- [ ] API REST
- [ ] Melhorias de performance

### Versão 1.2
- [ ] Suporte a tempo real
- [ ] Plugin para DAWs
- [ ] Modelos pré-treinados
- [ ] Interface mobile

### Versão 2.0
- [ ] Síntese de voz completa
- [ ] Controle emocional
- [ ] Suporte multilíngue
- [ ] Cloud deployment

---

**Desenvolvido com ❤️ para a comunidade de música**

*RVC Voice Cloning App v1.0 | Powered by RVC & Gradio*