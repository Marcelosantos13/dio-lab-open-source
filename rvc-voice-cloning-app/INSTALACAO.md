# 🎤 RVC Voice Cloning App - Guia de Instalação

## 📋 Pré-requisitos

- **Python 3.8 ou superior**
- **4GB de RAM** (recomendado 8GB)
- **2GB de espaço livre** em disco
- **Conexão com internet** (para instalação)

## 🚀 Instalação Rápida

### Windows

1. **Baixe e extraia** o arquivo ZIP do aplicativo
2. **Clique duas vezes** em `install_windows.bat`
3. **Aguarde** a instalação das dependências
4. **Execute** clicando em `executar.bat`

### Linux/Ubuntu

```bash
# Extrair arquivo
unzip rvc-voice-cloning-app.zip
cd rvc-voice-cloning-app

# Instalar
chmod +x install_linux.sh
./install_linux.sh

# Executar
./executar.sh
```

### macOS

```bash
# Extrair arquivo
unzip rvc-voice-cloning-app.zip
cd rvc-voice-cloning-app

# Instalar Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python
brew install python3

# Instalar dependências
./install_linux.sh

# Executar
./executar.sh
```

## 🔧 Instalação Manual

Se os scripts automáticos não funcionarem:

```bash
# 1. Verificar Python
python --version  # ou python3 --version

# 2. Instalar dependências
pip install -r requirements.txt  # ou pip3

# 3. Executar aplicativo
python main.py  # ou python3 main.py
```

## 🌐 Como Usar

1. **Execute** o aplicativo usando um dos métodos acima
2. **Aguarde** a mensagem "Aplicativo iniciado com sucesso!"
3. **O navegador abrirá automaticamente** em `http://localhost:12000`
4. **Se não abrir**, acesse manualmente: `http://localhost:12000`

## 📁 Estrutura de Arquivos

```
rvc-voice-cloning-app/
├── main.py              # Script principal
├── app.py               # Interface Gradio
├── rvc_core.py          # Modelo RVC
├── utils.py             # Utilitários
├── config.py            # Configurações
├── requirements.txt     # Dependências
├── install_windows.bat  # Instalador Windows
├── install_linux.sh     # Instalador Linux/Mac
├── executar.bat         # Executar Windows
├── executar.sh          # Executar Linux/Mac
├── examples/            # Arquivos de exemplo
└── README.md            # Documentação completa
```

## 🎵 Primeiros Passos

### 1. Conversão Básica
- Vá para aba **"Conversão de Voz"**
- Faça upload de um arquivo de áudio
- Ajuste os parâmetros conforme desejado
- Clique em **"Converter Voz"**

### 2. Formatos Suportados
- **Entrada**: WAV, MP3, FLAC, M4A, OGG
- **Saída**: WAV, MP3, FLAC
- **Qualidade**: Mínimo 16kHz, recomendado 44.1kHz

### 3. Dicas Importantes
- Use áudio **sem ruído de fundo**
- Duração ideal: **10 segundos a 10 minutos**
- Para melhores resultados: **áudio de alta qualidade**

## ⚙️ Configurações Avançadas

### Ajuste de Tom
- **Range**: -12 a +12 semitons
- **+3 a +6**: Feminizar voz
- **-3 a -6**: Masculinizar voz

### Ajuste de Formante
- **Range**: -1.0 a +1.0
- **Positivo**: Voz mais brilhante
- **Negativo**: Voz mais grave

## 🚨 Solução de Problemas

### Erro: "Python não encontrado"
```bash
# Windows
# Baixe de: https://python.org/downloads/

# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip

# macOS
brew install python3
```

### Erro: "Módulo não encontrado"
```bash
# Reinstalar dependências
pip install -r requirements.txt --upgrade
```

### Erro: "Porta em uso"
- Feche outros aplicativos que usem a porta 12000
- Ou edite `main.py` e mude `server_port=12000` para outro número

### Aplicativo não abre no navegador
- Acesse manualmente: `http://localhost:12000`
- Verifique se o firewall não está bloqueando

## 📞 Suporte

### Logs de Erro
- **Windows**: Verifique a janela do prompt de comando
- **Linux/Mac**: Verifique o terminal

### Informações do Sistema
```bash
# Verificar versões
python --version
pip --version
pip list | grep torch
pip list | grep gradio
```

### Contato
- **Issues**: Reporte problemas no GitHub
- **Documentação**: Consulte README.md completo
- **Comunidade**: Participe das discussões

## 🔄 Atualizações

Para atualizar o aplicativo:

1. **Baixe** a nova versão
2. **Substitua** os arquivos antigos
3. **Execute** o instalador novamente
4. **Mantenha** seus modelos personalizados

## 🎯 Performance

### Requisitos Mínimos
- **CPU**: Dual-core 2GHz
- **RAM**: 4GB
- **GPU**: Opcional (acelera processamento)

### Requisitos Recomendados
- **CPU**: Quad-core 3GHz+
- **RAM**: 8GB+
- **GPU**: NVIDIA com CUDA (para melhor performance)

## 📝 Licença

Este projeto está sob licença MIT. Veja LICENSE para detalhes.

---

**🎤 RVC Voice Cloning App v1.0**  
*Desenvolvido com ❤️ para a comunidade de música*