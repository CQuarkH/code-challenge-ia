import logging
import os

# aseguramos que exista la carpeta de logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

def get_logger(name: str):
    """
    Crea un logger configurado para escribir en archivo y no en consola.
    """
    # crear o recuperar el logger
    logger = logging.getLogger(name)
    
    # si ya tiene handlers (para evitar duplicar líneas si se llama varias veces), retornamos
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG) # capturamos todo desde DEBUG hacia arriba

    # configurar el handler de archivo (FileHandler)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # formato del log (timestamp - módulo - nivel - mensaje)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # añadir el handler al logger
    logger.addHandler(file_handler)
    
    return logger