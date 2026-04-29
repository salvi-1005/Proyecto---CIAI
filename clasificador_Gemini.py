"""
Proyecto: Derecho a las Víctimas
Script: Prueba de Gemini via API de Google

Descripción:
    Permite clasificar normativas legales usando distintos modelos de lenguaje.
    El script toma el archivo test.xlsx (sin etiquetas) y genera un Excel
    con las predicciones de cada clasificador para luego evaluar con evaluacion.ipynb.

Clasificadores a probar:
    - claude   : Claude via API de Anthropic (requiere API key)
    - gpt      : GPT via API de OpenAI (requiere API key)
    - gemini   : Gemini via API de Google (requiere API key — tier gratuito disponible)
    - groq     : Llama/Mixtral via Groq API (requiere API key — tier gratuito disponible)
    - ollama   : Modelos locales via Ollama (gratis, requiere Ollama instalado)
    - beto     : BETO zero-shot via HuggingFace (gratis, sin API key)

Uso:
    1. Completar la sección de configuración con las rutas y API keys
    2. Elegir el clasificador en CLASIFICADOR_ACTIVO
    3. Correr: python clasificador.py

Output:
    resultado_{CLASIFICADOR_ACTIVO}.xlsx — para evaluar con evaluacion.ipynb
"""

import pandas as pd
import os
import time

from dotenv import load_dotenv
load_dotenv()

# Configuración 

# Archivo de entrada (sin etiquetas)

FILE_TEST = 'test.xlsx'

# Clasificador a usar — cambiar según lo que se quiera probar:
CLASIFICADOR= 'gemini'
#API KEY
GOOGLE_API_KEY  = os.getenv('GOOGLE_API_KEY', '')       


MODELO = {
    'gemini': 'gemini-1.5-flash-8b',             # gratuito
}

# Pausa entre llamadas a la API (segundos) — evita rate limits
PAUSA_ENTRE_LLAMADAS = 5

# Prompt de clasificación 
# Basado en el criterio definido por los investigadores

PROMPT_SISTEMA = """Sos un experto en análisis de normativa legal argentina. 
Tu tarea es clasificar si una norma consagra o no derechos de las víctimas.

CRITERIO:
- caso_ok = 1 (SÍ consagra derechos de víctimas): la norma establece derechos, garantías u obligaciones 
  relativas a personas consideradas víctimas. Incluye mecanismos de denuncia, acompañamiento, asistencia, 
  reparación/indemnización, o define organigramas de organismos directamente vinculados a derechos de víctimas.

- caso_ok = 0 (NO consagra derechos de víctimas): la norma menciona la palabra "víctima" pero legisla 
  principalmente sobre otros sujetos (victimarios, operadores judiciales, testigos), o solo reorganiza 
  estructuras administrativas, o usa el término como contexto sin involucrar directamente a las víctimas.

EJEMPLOS:
- Ley 1487 (Subsidio a víctimas de inundaciones) → 1. Otorga asistencia directa a víctimas.
- Decreto 9088 (Subsidio a deudos de víctimas de accidente) → 1. Reparación económica directa.
- Decreto 558 (Pago de sentencia CIDH a víctimas) → 1. Ejecuta reparación concreta a víctimas.
- Ley 340 (Código Civil) → 0. Legisla sobre responsabilidad civil general, no sobre víctimas.
- Decreto 851 (Organigrama Programa Verdad y Justicia) → 0. Solo define estructura administrativa.
- Ley 25633 (Día Nacional de la Memoria) → 0. Declaración conmemorativa sin derechos concretos.

INSTRUCCIÓN:
Respondé con el número 1 o 0 seguido de dos puntos y una oración breve explicando tu decisión. Ejemplo: '1: La norma establece asistencia directa a víctimas de inundación."""

def construir_prompt_usuario(row):
    """Construye el prompt con los datos de la norma."""
    titulo  = str(row.get('Título', '')).strip()  if pd.notna(row.get('Título'))  else ''
    resumen = str(row.get('Resumen', '')).strip() if pd.notna(row.get('Resumen')) else ''
    articulos = str(row.get('Artículos', '')).strip() if pd.notna(row.get('Artículos')) else ''

    texto = f"Tipo: {row.get('Tipo', '')}\nNúmero: {row.get('Número', '')}\nTítulo: {titulo}"
    if resumen:
        texto += f"\nResumen: {resumen}"
    if articulos:
        texto += f"\nArtículos relevantes: {articulos[:500]}"  # limita para no exceder tokens

    texto += "\n\n¿Esta norma consagra derechos de las víctimas? Respondé solo con 1 o 0."
    return texto


def parsear_respuesta(respuesta):
    respuesta = str(respuesta).strip()
    if respuesta.startswith('1'):
        return 1, respuesta
    elif respuesta.startswith('0'):
        return 0, respuesta
    else:
        return -1, f"Respuesta inválida: {respuesta}"

# Clasificadores 


def clasificar_con_gemini(df):
    try:
        from google import genai
    except ImportError:
        os.system('pip install google-genai -q')
        from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    predicciones = []
    justificaciones = []

    for i, row in df.iterrows():
        prompt = construir_prompt_usuario(row)
        try:
            response = client.models.generate_content(
                model=MODELO['gemini'],
                contents=PROMPT_SISTEMA + '\n\n' + prompt
            )
            respuesta = response.text
            pred, just = parsear_respuesta(respuesta)
        except Exception as e:
            pred = -1
            just = f"Error: {e}"
        predicciones.append(pred)
        justificaciones.append(just)
        print(f"  [{i+1}/{len(df)}] Norma {row.get('Número', '')} → {pred}")
        time.sleep(PAUSA_ENTRE_LLAMADAS)

    return predicciones, justificaciones


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print(f'  Clasificador: {CLASIFICADOR.upper()}')
    print('=' * 60)

    # Cargar datos
    print(f'\nCargando {FILE_TEST}...')
    df = pd.read_excel(FILE_TEST)
    df['Número'] = df['Número'].astype(str).str.strip()
    print(f'  {len(df)} normas cargadas')

    # Clasificar
    print(f'\nClasificando con {CLASIFICADOR}...')


    clasificadores = {
        'gemini': clasificar_con_gemini,
    }
    predicciones, justificaciones = clasificadores[CLASIFICADOR](df)

    # Armar resultado
    df_resultado = df[['Tipo', 'Número', 'Título']].copy()
    df_resultado['caso_ok'] = predicciones
    df_resultado['Justificación'] = justificaciones

    # Estadísticas
    validos = [p for p in predicciones if p != -1]
    errores = [p for p in predicciones if p == -1]
    print(f'\n── Resultado ──────────────────────────────────────')
    print(f'  Total clasificadas: {len(validos)} de {len(df)}')
    print(f'  caso_ok = 1: {predicciones.count(1)}')
    print(f'  caso_ok = 0: {predicciones.count(0)}')
    if errores:
        print(f'  Errores:     {len(errores)} (revisar columna Justificación)')

    # Guardar
    output = f'resultado_{CLASIFICADOR}.xlsx'
    df_resultado.to_excel(output, index=False)
    print(f'\n Guardado: {output}')
if __name__ == '__main__':
    main()
