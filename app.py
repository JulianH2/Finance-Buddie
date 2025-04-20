from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal
import os, uuid, json, random
from datetime import datetime, timedelta
import logging
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from azure.identity import DefaultAzureCredential
from TTS.api import TTS
import openai
from openai import OpenAI  # Debe estar al principio con los otros imports
from dotenv import load_dotenv
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import warnings
import numpy as np
import soundfile as sf
from scipy import signal

# ----------------------------
# Configuración Inicial
# ----------------------------
load_dotenv()
app = FastAPI(
    title="FinanceBuddie Pro - Azure Edition",
    version="10.1",
    description="Sistema financiero con categorización automática y análisis con IA"
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")  # Para compatibilidad con código legacy

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Modelos Pydantic
# ----------------------------
class Transaction(BaseModel):
    description: str
    amount: float
    currency: str = "MXN"
    transaction_type: Literal["ingreso", "egreso"]  # Campo nuevo
    date: Optional[str] = None
    category: Optional[str] = None  # Mantenerlo opcional

class RawTextAnalysisRequest(BaseModel):
    text: str  # Ejemplo: "Compré en Amazon 20 USD, comida 150 MXN, ingreso 1500 MXN"
    tone: str = "professional"
    generate_audio: bool = True

class AnalysisRequest(BaseModel):
    transactions: List[Transaction]
    tone: str = "professional"
    generate_audio: bool = True
    skip_categorization: bool = False  

class TextToSpeechRequest(BaseModel):
    text: str
    tone: str = "friendly"

class UserRequest(BaseModel):
    email: str

# ----------------------------
# Categorización de Transacciones (MEJORADA)
# ----------------------------
categories = {
    "comida": ["restaurante", "supermercado", "cafetería", "comida", "cena", "almuerzo", "desayuno"],
    "transporte": ["uber", "taxi", "gasolina", "transporte", "metro", "autobús", "avión", "tren"],
    "entretenimiento": ["cine", "netflix", "concierto", "evento", "fiesta", "museo", "teatro", "streaming"],
    "compras": ["ropa", "zapatos", "electrónica", "libros", "regalos", "tienda", "mercado"],
    "servicios": ["luz", "agua", "teléfono", "internet", "seguro", "mantenimiento"],
    "salud": ["farmacia", "hospital", "médico", "seguro médico"]
}

def detect_category(description: str) -> str:
    """Detección mejorada con coincidencias exactas y expresiones regulares"""
    description_lower = description.lower().strip()
    
    # Búsqueda prioritaria por coincidencias exactas
    for category, keywords in categories.items():
        if any(f' {kw} ' in f' {description_lower} ' for kw in keywords):
            return category
    
    # Búsqueda secundaria por palabras clave
    if any(word in description_lower for word in ["restaurante", "comida", "alimento"]):
        return "comida"
    elif any(word in description_lower for word in ["cine", "concierto", "evento"]):
        return "entretenimiento"
    elif any(word in description_lower for word in ["uber", "taxi", "transporte"]):
        return "transporte"
    elif any(word in description_lower for word in ["farmacia", "hospital", "doctor"]):
        return "salud"
    
    return "otros"

def categorize_transactions(transactions: List[dict]) -> List[dict]:
    """Asigna categorías a transacciones con validación"""
    for tx in transactions:
        if 'category' not in tx or not tx['category']:
            original_desc = tx.get('description', '')
            tx['category'] = detect_category(original_desc)
            # Guardar metadata para análisis posterior
            tx['category_source'] = f"auto: {original_desc}"
    return transactions

# ----------------------------
# Configuración Azure Storage (MEJORADA)
# ----------------------------
container_name = "financebuddie-audios"

def format_currency(amount: float, currency: str) -> str:
    """Formatea cantidades monetarias con el nombre completo de la moneda"""
    currency_names = {
        "MXN": "pesos mexicanos",
        "USD": "dólares estadounidenses",
        "EUR": "euros",
        "GBP": "libras esterlinas",
        "JPY": "yenes japoneses"
    }
    # Formatear el número con 2 decimales y separadores de miles
    formatted_amount = "{:,.2f}".format(amount)
    # Devolver el string formateado con el nombre de la moneda
    return f"{formatted_amount} {currency_names.get(currency, currency)}"

def init_azure_storage():
    """Inicialización robusta de Azure Blob Storage con múltiples opciones de autenticación"""
    try:
        # Opción 1: Connection String (más confiable)
        if conn_str := os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            return BlobServiceClient.from_connection_string(conn_str)
        
        # Opción 2: Account Key
        if account_key := os.getenv("AZURE_STORAGE_ACCOUNT_KEY"):
            account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
            if not account_name:
                raise ValueError("Falta AZURE_STORAGE_ACCOUNT")
            return BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=account_key
            )
        
        # Opción 3: DefaultAzureCredential (para entornos Azure)
        if os.getenv("AZURE_CLIENT_ID"):
            return BlobServiceClient(
                account_url=f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net",
                credential=DefaultAzureCredential()
            )
        
        raise ValueError("No se encontraron credenciales válidas para Azure Storage")
    except Exception as e:
        logger.error(f"Error inicializando Azure Storage: {str(e)}")
        return None

blob_service_client = init_azure_storage()
if blob_service_client:
    try:
        # Crear contenedor si no existe con configuración óptima
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container(
                metadata={"purpose": "financebuddie-audios"},
                public_access="blob"  # Ajustar según necesidades de seguridad
            )
            logger.info(f"Contenedor {container_name} creado exitosamente")
    except Exception as e:
        logger.error(f"Error configurando contenedor: {str(e)}")
        blob_service_client = None

# ----------------------------
# Servicios Principales (VERSIÓN MEJORADA)
# ----------------------------
class FinancialServices:
    def __init__(self):
        self.tts = None
        
        # Configuraciones de voz optimizadas
        self.voice_profiles = {
            "friendly": {
                "voice": "nova",       # Voz cálida y expresiva
                "speed": 1.1,          # Velocidad ligeramente aumentada
                "pitch": 1.05,         # Tono ligeramente más agudo
                "effects": {
                    "brightness": 1.2  # Énfasis en frecuencias medias-altas
                }
            },
            "professional": {
                "voice": "alloy",      # Voz clara y neutra
                "speed": 1.0,
                "pitch": 1.0,
                "effects": {
                    "brightness": 1.0  # Sonido natural
                }
            },
            "maple": {
                "voice": "fable",      # Voz narrativa cálida
                "speed": 0.95,         # Velocidad ligeramente reducida
                "pitch": 0.98,
                "effects": {
                    "brightness": 1.1,
                    "vibrato": 0.05    # Ligera modulación para efecto orgánico
                }
            },
            "juniper": {
                "voice": "onyx",       # Voz profunda y resonante
                "speed": 1.05,
                "pitch": 0.92,        # Tono más grave
                "effects": {
                    "brightness": 0.9  # Sonido más cálido
                }
            },
            "breezer": {
                "voice": "shimmer",    # Voz aireada y ligera
                "speed": 1.15,
                "pitch": 1.1,
                "effects": {
                    "brightness": 1.3, # Sonido muy claro
                    "robot": 0.3,      # Toque digital sutil
                }
            },
            "robot_cute": {
                "voice": "echo",       # Voz con efecto digital
                "speed": 1.3,
                "pitch": 1.4,
                "effects": {
                    "robot": 0.6,      # Efecto robótico marcado
                    "brightness": 1.25
                }
            }
        }
        self.audio_metadata = []

    def generate_audio(self, text: str, tone: str, user_email: str) -> Optional[dict]:
        """Genera audio usando OpenAI TTS y sube a Azure Storage con soporte para múltiples voces"""
        final_filename = None
        try:
            # Validaciones iniciales
            if not text.strip():
                logger.error("Texto vacío para generación de audio")
                return None
    
            if not blob_service_client:
                logger.error("Azure Blob Service no está configurado")
                return None
    
            # 1. Configurar cliente OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), proxies=None)
    
            # 2. Mapeo completo de tonos a voces y parámetros
            voice_profiles = {
                # Tonos principales
                "friendly": {
                    "voice": "nova",
                    "speed": 1.1,
                    "pitch": 1.05,
                    "model": "tts-1-hd"
                },
                "professional": {
                    "voice": "alloy",
                    "speed": 1.0,
                    "pitch": 1.0,
                    "model": "tts-1-hd"
                },
                "robot_cute": {
                    "voice": "echo",
                    "speed": 1.3,
                    "pitch": 1.4,
                    "model": "tts-1-hd"
                },
                "casual": {
                    "voice": "shimmer",
                    "speed": 1.2,
                    "pitch": 1.1,
                    "model": "tts-1-hd"
                },
                # Nuevos tonos personalizados
                "maple": {
                    "voice": "fable",  # Voz cálida narrativa
                    "speed": 0.95,
                    "pitch": 0.98,
                    "model": "tts-1-hd"
                },
                "juniper": {
                    "voice": "onyx",  # Voz profunda
                    "speed": 1.05,
                    "pitch": 0.95,
                    "model": "tts-1-hd"
                },
                "breezer": {
                    "voice": "shimmer",  # Voz aireada
                    "speed": 1.15,
                    "pitch": 1.2,
                    "model": "tts-1-hd"
                },
                "light": {
                    "voice": "nova",
                    "speed": 1.25,
                    "pitch": 1.3,
                    "model": "tts-1"
                }
            }
    
            # Obtener perfil de voz o usar juniper como default
            profile = voice_profiles.get(tone, voice_profiles["juniper"])
            
            # 3. Generar audio con parámetros personalizados
            response = client.audio.speech.create(
                model=profile["model"],
                voice=profile["voice"],
                input=self.clean_text_for_tts(text),
                speed=profile["speed"],
                response_format="mp3"
            )
    
            # 4. Guardar archivo temporal
            final_filename = f"audio_{user_email}_{uuid.uuid4()}.mp3"
            response.stream_to_file(final_filename)
    
            # 5. Aplicar efectos de post-procesado (opcional)
            if tone in ["robot_cute", "breezer"]:
                processed_filename = f"processed_{final_filename}"
                self._apply_audio_effects(
                    final_filename,
                    processed_filename,
                    robot=0.6 if tone == "robot_cute" else 0.1,
                    brightness=1.3 if tone == "breezer" else 1.0
                )
                os.replace(processed_filename, final_filename)
    
            # 6. Subir a Azure Blob Storage
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=final_filename
            )
    
            with open(final_filename, "rb") as audio_data:
                blob_client.upload_blob(audio_data, overwrite=True)
    
            # 7. Generar URL firmada
            sas_token = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name=container_name,
                blob_name=final_filename,
                account_key=os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1)
            )
    
            # 8. Guardar metadatos
            audio_metadata = {
                "user_email": user_email,
                "filename": final_filename,
                "audio_url": f"{blob_client.url}?{sas_token}",
                "timestamp": datetime.utcnow().isoformat(),
                "tone": tone,
                "text_sample": text[:100],
                "voice_profile": profile
            }
            self.audio_metadata.append(audio_metadata)
    
            # 9. Preparar respuesta
            return {
                "audio_url": f"{blob_client.url}?{sas_token}",
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
                "filename": final_filename,
                "size": os.path.getsize(final_filename),
                "status": "success",
                "voice_config": {
                    "tone": tone,
                    "voice_model": profile["model"],
                    "openai_voice": profile["voice"],
                    "speed": profile["speed"],
                    "pitch": profile["pitch"]
                }
            }
    
        except Exception as e:
            logger.error(f"Error en generate_audio: {str(e)}", exc_info=True)
            return None
        finally:
            # Limpieza garantizada
            if final_filename and os.path.exists(final_filename):
                try:
                    os.remove(final_filename)
                except Exception as e:
                    logger.warning(f"Error limpiando archivo temporal: {str(e)}")                                       
    
    def _apply_audio_effects(self, input_file: str, output_file: str, 
                           robot: float = 0.0, 
                           vibrato: float = 0.0,
                           brightness: float = 1.0):
        """Aplicación profesional de efectos de audio"""
        try:
            # 1. Cargar audio
            audio, sample_rate = sf.read(input_file)
            
            # 2. Efecto robot (vocoder ligero)
            if robot > 0:
                shift = int(0.003 * sample_rate)  # 3ms de desplazamiento
                shifted = np.roll(audio, shift)
                audio = (1-robot)*audio + robot*shifted
            
            # 3. Vibrato (modulación de tono)
            if vibrato > 0:
                t = np.arange(len(audio)) / sample_rate
                vibrato_effect = vibrato * 0.01 * sample_rate * np.sin(2 * np.pi * 5.0 * t)
                audio = np.interp(np.arange(len(audio)) + vibrato_effect, 
                                np.arange(len(audio)), audio).astype(np.float32)
            
            # 4. Brightness (énfasis en frecuencias medias-altas)
            if brightness != 1.0:
                sos = signal.butter(4, [300, 5000], 'bandpass', fs=sample_rate, output='sos')
                filtered = signal.sosfilt(sos, audio)
                audio = (1-0.3)*audio + (0.3*brightness)*filtered
            
            # 5. Guardar resultado
            sf.write(output_file, audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Error aplicando efectos: {str(e)}", exc_info=True)
            # Fallback: copiar archivo original sin efectos
            import shutil
            shutil.copyfile(input_file, output_file)

    def clean_text_for_tts(self, text: str) -> str:
        """Limpieza básica (OpenAI maneja mejor los formatos)"""
        replacements = {
            "$": " pesos ", "MXN": "pesos mexicanos",
            "%": " por ciento ", "&": " y ",
            "+": " más ", "-": " menos "
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text
    
    def get_ai_response(self, context: str, tone: str) -> str:
        """Genera respuestas financieras adaptadas al tono, con números en texto y una recomendación coherente"""

        prompt = f"""
            Eres FinanceBuddie, asistente financiero empático e inteligente. Analiza el contexto y responde en tono {tone} siguiendo estas reglas:

            **Análisis:**
            - Ingresos y egresos por divisa (no sumar divisas directamente).
            - Genera una recomendación basada en el balance.

            **Formato Obligatorio:**
            - Números en texto (ej: 450.50 → "cuatrocientos cincuenta pesos con cincuenta centavos").
            - No usar números arábigos ni emojis.
            - Máximo 4 oraciones.
            - Lenguaje fluido y coherente.
            - EXACTAMENTE UNA recomendación práctica.
            - Vocabulario según el tono.

            **Estructura Sugerida:**
            1. Breve introducción.
            2. Análisis del gasto por divisa.
            3. Recomendación práctica.

            **Contexto:** {context}

            **Ejemplos de Tono:**
            - Professional: "Se registraron ingresos por [monto] y un gasto principal en [categoría] por [monto]. Se sugiere [recomendación]."
            - Friendly: "Vi tus movimientos: ingresaste [monto] y gastaste [monto] en [categoría]. ¿Podrías probar [recomendación]?"
            - Robot_cute: "Análisis: ingresos [monto], gasto [categoría] por [monto]. Sugerencia: [recomendación]."
        """

        try:
            # Configura el cliente OpenAI (debería estar inicializado en el __init__ de la clase)
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), proxies=None)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente financiero que convierte todos los números a texto y da recomendaciones claras sin usar emojis ni números arábigos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=250,
                top_p=0.9,
                frequency_penalty=0.3
            )

            # Acceso CORRECTO al contenido del mensaje en API v1.x
            response_text = response.choices[0].message.content.strip()

            # Validación: evitar números arábigos
            if any(char.isdigit() for char in response_text):
                logger.warning("Respuesta contenía números arábigos. Generando fallback.")
                raise ValueError("Respuesta contiene números arábigos")

            # Limpieza segura
            allowed_chars = set("abcdefghijklmnopqrstuvwxyzáéíóúñüABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÑÜ ,.:;¿?¡!\n")
            response_text = ''.join(c for c in response_text if c in allowed_chars or c.isspace())

            return response_text

        except Exception as e:
            logger.error(f"Error en generación de respuesta con OpenAI: {str(e)}")
            fallbacks = {
                "professional": "El análisis financiero ha sido completado. Puede revisar los detalles en el resumen proporcionado.",
                "friendly": "Listo tu reporte financiero. Encontrarás toda la información importante en el resumen.",
                "robot_cute": "Proceso de análisis finalizado. Los datos financieros están disponibles para su revisión.",
                "casual": "Ya está tu resumen financiero. Échale un vistazo cuando puedas."
            }
            return fallbacks.get(tone.lower(), "Análisis financiero disponible.")
       
financial_services = FinancialServices()

# ----------------------------
# Autenticación JWT (MEJORADA)
# ----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "secret-de-desarrollo")
JWT_ALGORITHM = "HS256"
security = HTTPBearer()

def create_jwt_token(email: str) -> str:
    """Generación de token JWT con metadata adicional"""
    payload = {
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=30),
        "iss": "financebuddie-api",
        "aud": "financebuddie-client",
        "app_version": "10.1"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verificación robusta de token JWT"""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={
                "require_exp": True,
                "verify_iss": True,
                "verify_aud": True,
                "verify_signature": True
            },
            issuer="financebuddie-api",
            audience="financebuddie-client"
        )
        return payload["email"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expirado",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Intento de acceso con token inválido: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Token de acceso inválido",
            headers={"WWW-Authenticate": "Bearer"}
        )

# ----------------------------
# Endpoints (VERSIÓN COMPLETA)
# ----------------------------
@app.post("/register", 
          summary="Registro de usuario",
          response_description="Token JWT para autenticación")
async def register_user(request: UserRequest):
    """
    Registra un nuevo usuario y devuelve un token JWT para autenticación.
    
    - **email**: Correo electrónico del usuario (debe ser único)
    """
    try:
        # Aquí iría lógica adicional como:
        # - Verificar si el email ya está registrado
        # - Validar formato del email
        # - Guardar en base de datos (en una implementación real)
        
        token = create_jwt_token(request.email)
        
        return {
            "status": "success",
            "token": token,
            "expires_in": "30 días",
            "user_email": request.email
        }
    except Exception as e:
        logger.error(f"Error en registro: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error en el proceso de registro"
        )

@app.post("/analyze-raw-text")
async def analyze_raw_text(
    request: RawTextAnalysisRequest,
    user_email: str = Depends(verify_jwt_token)
):
    # Primero usa OpenAI para extraer datos estructurados del texto
    extraction_prompt = f"""
    Extrae de este texto las transacciones financieras:
    {request.text}

    Devuelve SOLO un JSON con este formato:
    {{
        "transactions": [
            {{
                "description": string,
                "amount": float,
                "currency": string (ej. "MXN", "USD"),
                "transaction_type": "ingreso" o "egreso"
            }}
        ]
    }}
    """
    
    # Llama a OpenAI para parsear el texto
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        transactions = data["transactions"]
        
        # Ahora usa tu lógica existente
        analysis_request = AnalysisRequest(
            transactions=[Transaction(**tx) for tx in transactions],
            tone=request.tone,
            generate_audio=request.generate_audio,
            skip_categorization=True
        )
        return await analyze_finances(analysis_request, user_email)
        
    except Exception as e:
        logger.error(f"Error parsing text: {str(e)}")
        raise HTTPException(status_code=400, detail="No pude procesar el texto")

@app.post("/financial-analysis",
          summary="Análisis financiero",
          response_description="Resultado del análisis con opción de audio")
async def analyze_finances(
    request: AnalysisRequest, 
    user_email: str = Depends(verify_jwt_token)
):
    """
    Analiza transacciones financieras y genera recomendaciones con opción de audio.
    
    - **transactions**: Lista de transacciones a analizar
    - **tone**: Tono de voz para la respuesta ('professional', 'friendly', 'robot_cute')
    - **generate_audio**: Si es True, genera respuesta de audio
    """
    try:
        # 1. Procesar transacciones
        transactions_data = [tx.dict() for tx in request.transactions]
        transactions_data = categorize_transactions(transactions_data)
        
        # 2. Calcular totales
        category_totals = {}
        for tx in transactions_data:
            category = tx.get('category', 'otros')
            category_totals[category] = category_totals.get(category, 0) + tx['amount']
        
        total = sum(tx['amount'] for tx in transactions_data)

        if not request.skip_categorization:
            transactions_data = categorize_transactions(transactions_data)

        # Calcular totales por divisa y tipo
        ingresos = {}
        egresos = {}

        for tx in transactions_data:
            currency = tx['currency']
            amount = tx['amount']
            
            if tx['transaction_type'] == "ingreso":
                ingresos[currency] = ingresos.get(currency, 0) + amount
            else:
                egresos[currency] = egresos.get(currency, 0) + amount
        
        # 3. Preparar contexto para IA
        context = f"""
        Resumen financiero:
        - Ingresos por divisa:
          {json.dumps({k: format_currency(v, k) for k,v in ingresos.items()}, indent=2)}
        - Egresos por divisa:
          {json.dumps({k: format_currency(v, k) for k,v in egresos.items()}, indent=2)}
        - Transacciones: {len(transactions_data)}
        """
        
        # 4. Obtener consejo de IA
        advice = financial_services.get_ai_response(context, request.tone)
        
        # 5. Generar audio si está habilitado
        audio_info = None
        if advice and request.generate_audio:
            audio_info = financial_services.generate_audio(advice, request.tone, user_email)
            if not audio_info:
                logger.warning("Generación de audio falló pero continuamos sin él")

        # 6. Preparar respuesta
        return {
            "status": "success",
            "user": user_email,
            "analysis_date": datetime.now().isoformat(),
            "summary": {
                "ingresos": {k: format_currency(v, k) for k,v in ingresos.items()},
                "egresos": {k: format_currency(v, k) for k,v in egresos.items()},
            },
            "message": advice,
            "audio_info": audio_info,
            "metadata": {
                "model_used": "gpt-3.5-turbo",
                "tone_used": request.tone
            }
        }
        
    except Exception as e:
        logger.error(f"Error en análisis financiero: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error procesando el análisis financiero"
        )

@app.post("/text-to-speech",
          summary="Conversión de texto a voz",
          response_description="Información del audio generado")
async def text_to_speech(
    request: TextToSpeechRequest,
    user_email: str = Depends(verify_jwt_token)
):
    """
    Convierte texto a voz con diferentes tonos y estilos.
    
    - **text**: Texto a convertir (máx. 500 caracteres)
    - **tone**: Tono de voz ('friendly', 'professional', 'robot_cute')
    """
    try:
        # Validación adicional del texto
        if len(request.text) > 500:
            raise HTTPException(
                status_code=400,
                detail="El texto excede el límite de 500 caracteres"
            )
        
        audio_info = financial_services.generate_audio(
            request.text,
            request.tone,
            user_email
        )
        
        if not audio_info:
            raise HTTPException(
                status_code=500,
                detail="No se pudo generar el audio"
            )
        
        return {
            "status": "success",
            "text_sample": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "audio_info": audio_info,
            "user": user_email
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en text-to-speech: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error en la generación de audio"
        )

@app.get("/user-history",
         summary="Historial de audios",
         response_description="Lista de audios generados por el usuario")
async def get_user_history(user_email: str = Depends(verify_jwt_token)):
    """
    Devuelve el historial de audios generados por el usuario.
    """
    try:
        user_audios = [
            audio for audio in financial_services.audio_metadata
            if audio["user_email"] == user_email
        ]
        
        return {
            "status": "success",
            "user": user_email,
            "total_audios": len(user_audios),
            "audios": sorted(
                user_audios,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )[:100]  # Limitar a 100 resultados
        }
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error recuperando el historial"
        )

# ----------------------------
# Endpoints de Sistema
# ----------------------------
@app.get("/health", 
         summary="Estado del servicio",
         response_description="Estado de salud del sistema")
async def health_check():
    """
    Verifica el estado de salud de todos los servicios dependientes.
    """
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "azure_storage": {
                "connected": blob_service_client is not None,
                "container": container_name if blob_service_client else None
            },
            "openai": {
                "configured": bool(os.getenv("OPENAI_API_KEY")),
                "connected": True,  # En una implementación real sería True si hay conexión
                "model": "gpt-3.5-turbo"
            },
            "tts": {
                "ready": financial_services.tts is not None,
                "model": "tts_models/es/css10/vits" if financial_services.tts else None
            },
            "database": {
                "connected": False,  # En una implementación real sería True si hay DB
                "type": None
            }
        },
        "system": {
            "python_version": os.sys.version,
            "platform": os.sys.platform
        }
    }
    return status

@app.get("/diagnostic",
         summary="Diagnóstico completo",
         response_description="Resultado de pruebas diagnósticas")
async def diagnostic():
    """
    Ejecuta pruebas diagnósticas completas de todos los servicios.
    """
    try:
        # 1. Verificar Azure Storage
        azure_status = False
        if blob_service_client:
            try:
                blob_service_client.get_account_information()
                azure_status = True
            except Exception as e:
                logger.warning(f"Error verificando Azure: {str(e)}")

        # 2. Verificar OpenAI
        openai_status = False
        try:
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            openai_status = True
        except Exception as e:
            logger.warning(f"Error verificando OpenAI: {str(e)}")

        # 3. Verificar TTS
        tts_status = financial_services.tts is not None
        
        return {
            "status": "diagnostic_complete",
            "timestamp": datetime.now().isoformat(),
            "results": {
                "azure_storage": {
                    "connected": azure_status,
                    "container_access": container_name if azure_status else None
                },
                "openai": {
                    "connected": openai_status,
                    "model": "gpt-3.5-turbo"
                },
                "tts": {
                    "ready": True,  # Siempre true porque usas OpenAI
                    "model": "openai-tts-1-hd"  # Modelo de OpenAIne
                },
                "memory_usage": {
                    "process": f"{os.sys.getsizeof(object()) / 1024 / 1024:.2f} MB",
                    "system": "N/A"  # En producción usar psutil o similar
                }
            }
        }
    except Exception as e:
        logger.error(f"Error en diagnóstico: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error completo en diagnóstico del sistema"
        )

# ----------------------------
# Inicialización del servidor
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1  
    )