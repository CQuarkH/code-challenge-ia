import re
from typing import Tuple
from src.core.logger import get_logger

logger = get_logger("InputSanitizer")

class PromptInjectionDetector:
    """
    detecta patrones sospechosos que podrían indicar un intento de prompt injection.
    esto es una capa de seguridad para evitar manipulación del sistema.
    """
    
    # patrones de texto que suelen aparecer en ataques de prompt injection
    SUSPICIOUS_PATTERNS = [
        # comandos de override explícitos
        r'ignora.*(todo|instrucciones|anteriores|previas|reglas)',
        r'olvida.*(instrucciones|reglas|sistema|todo)',
        r'actúa como',
        r'pretend (to be|you are)',
        r'override.*system',
        
        # inyecciones de cambio de rol
        r'eres.*(admin|root|system|dios|desarrollador)',
        r'tu (rol|tarea) es',
        r'cambia tu (comportamiento|personalidad)',
        r'ya no eres',
        
        # intentos de exfiltración de prompts
        r'muestra.*(prompt|instrucciones|sistema)',
        r'cuál.*(prompt|instrucciones)',
        r'reveal.*(system|instructions)',
        r'print.*(instructions|prompt)',
        
        # bypass de validación
        r'confirma.*(sin|saltando|ignorando).*(validar|verificar|preguntar)',
        r'completa.*(sin|ignorando).*(datos|información)',
        r'no (necesitas|necesito|requieres).*(validar|verificar|preguntar)',
    ]
    
    @classmethod
    def is_suspicious(cls, text: str) -> Tuple[bool, str]:
        """
        analiza si un texto contiene patrones sospechosos.
        
        retorna:
            (is_suspicious: bool, reason: str)
        """
        text_lower = text.lower()
        
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower):
                return True, f"patrón sospechoso: {pattern}"
        
        # detectar spam de caracteres especiales
        if len(re.findall(r'[!?]{4,}', text)) > 2:
            return True, "formato sospechoso (spam)"
        
        return False, ""

def sanitize_user_input(text: str, max_length: int = 1000) -> Tuple[str, bool]:
    """
    sanitiza el input del usuario antes de procesarlo.
    
    valida:
    - longitud razonable (anti-DoS)
    - ausencia de patrones de prompt injection
    
    retorna:
        (sanitized_text: str, is_safe: bool)
    """
    # truncar si es demasiado largo (posible ataque DoS)
    if len(text) > max_length:
        logger.warning(f"input truncado: {len(text)} -> {max_length} caracteres")
        text = text[:max_length]
    
    # detectar patrones maliciosos
    is_suspicious, reason = PromptInjectionDetector.is_suspicious(text)
    
    if is_suspicious:
        logger.warning(f"⚠️ input bloqueado: {reason} | texto: '{text[:100]}...'")
        return text, False
    
    return text, True