#!/usr/bin/env python3
"""
RVC Voice Cloning App - Executável Principal
"""

import sys
import os
import webbrowser
import time
import threading
from pathlib import Path

# Adicionar o diretório atual ao path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def open_browser():
    """Abre o navegador após um delay"""
    time.sleep(3)  # Aguarda 3 segundos para o servidor iniciar
    webbrowser.open('http://localhost:12000')

def main():
    """Função principal"""
    print("🎤 RVC Voice Cloning App")
    print("=" * 50)
    print("Iniciando aplicativo de clonagem de voz...")
    print("Aguarde alguns segundos...")
    print()
    
    try:
        # Importar e executar o app
        from app import create_interface
        
        # Criar interface
        app = create_interface()
        
        # Abrir navegador em thread separada
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("✅ Aplicativo iniciado com sucesso!")
        print("🌐 Abrindo navegador automaticamente...")
        print("📍 URL: http://localhost:12000")
        print()
        print("💡 Dicas:")
        print("   - Use arquivos de áudio de alta qualidade")
        print("   - Formatos suportados: WAV, MP3, FLAC")
        print("   - Para fechar, pressione Ctrl+C")
        print()
        print("🚀 Interface web carregando...")
        print("-" * 50)
        
        # Executar aplicativo
        app.launch(
            server_name="0.0.0.0",
            server_port=12000,
            share=False,
            show_error=True,
            debug=False,
            quiet=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Aplicativo encerrado pelo usuário.")
        print("Obrigado por usar o RVC Voice Cloning App!")
        
    except Exception as e:
        print(f"\n❌ Erro ao iniciar aplicativo: {e}")
        print("\nVerifique se todas as dependências estão instaladas.")
        print("Execute: pip install -r requirements.txt")
        input("\nPressione Enter para sair...")

if __name__ == "__main__":
    main()