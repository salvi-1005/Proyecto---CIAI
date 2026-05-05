"""
Proyecto: Derecho a las Víctimas

Descripción:
    Permite clasificar normativas legales usando distintos modelos de lenguaje.
    El script toma el archivo test.xlsx (sin etiquetas) y genera un Excel
    con las predicciones de cada clasificador para luego evaluar en el archivo de evaluacion.ipynb 
    con el test_con_etiquetas.xlsx

Clasificadores disponibles:
    - claude   : Claude via API de Anthropic (requiere API key)
    - gpt      : GPT via API de OpenAI (requiere API key)
    - gemini   : Gemini via API de Google (requiere API key — tier gratuito disponible)
    - groq     : Llama/Mixtral via Groq API (requiere API key — tier gratuito disponible)
    - ollama   : Modelos locales via Ollama (gratis, requiere Ollama instalado)
    - beto     : BETO zero-shot via HuggingFace (gratis, sin API key)
Output:
    resultado_{CLASIFICADOR_ACTIVO}.xlsx — para evaluar con evaluacion.ipynb
"""

import pandas as pd
import os

# Configuración 
FILE_TEST = 'test.xlsx'

# Clasificador a usar:
CLASIFICADOR_ACTIVO = 'ollama'

# Modelos específicos por proveedor
MODELOS = {
    'ollama': 'llama3.2',                    # modelo local
}

# Pausa entre llamadas a la API (segundos) — evita rate limits
PAUSA_ENTRE_LLAMADAS = 1

# Prompt de clasificación 
# Basado en el criterio definido por los investigadores

PROMPT_SISTEMA = """Sos un experto en análisis de normativa legal Argentina. 
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
Respondé ÚNICAMENTE con el número 1 o 0, sin explicación ni texto adicional."""

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
    """Extrae 0 o 1 de la respuesta del modelo."""
    respuesta = str(respuesta).strip()
    if '1' in respuesta[:10]:
        return 1
    elif '0' in respuesta[:10]:
        return 0
    else:
        return -1  # indica respuesta inválida


def clasificar_con_ollama(df):
    import requests

    predicciones = []
    justificaciones = []

    for i, row in df.iterrows():
        prompt = construir_prompt_usuario(row)
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': MODELOS['ollama'],
                    'prompt': PROMPT_SISTEMA + '\n\n' + prompt,
                    'stream': False
                }
            )
            respuesta = response.json()['response']
            pred = parsear_respuesta(respuesta)
            just = respuesta[:200]
        except Exception as e:
            pred = -1
            just = f"Error: {e}"
        predicciones.append(pred)
        justificaciones.append(just)
        print(f"  [{i+1}/{len(df)}] Norma {row.get('Número', '')} → {pred}")

    return predicciones, justificaciones

# Main 
def main():
    print('=' * 60)
    print(f'  Clasificador: {CLASIFICADOR_ACTIVO.upper()}')
    print('=' * 60)

    # Cargar datos
    print(f'\nCargando {FILE_TEST}...')
    df = pd.read_excel(FILE_TEST)
    df['Número'] = df['Número'].astype(str).str.strip()
    print(f'  {len(df)} normas cargadas')

    # Clasificar
    print(f'\nClasificando con {CLASIFICADOR_ACTIVO}...')
    clasificadores = {
        'ollama': clasificar_con_ollama,
    }

    if CLASIFICADOR_ACTIVO not in clasificadores:
        print(f'Error: clasificador "{CLASIFICADOR_ACTIVO}" no reconocido.')
        print(f'Opciones: {list(clasificadores.keys())}')
        return

    predicciones, justificaciones = clasificadores[CLASIFICADOR_ACTIVO](df)

    # Armar resultado
    df_resultado = df[['Tipo', 'Número', 'Título']].copy()
    df_resultado['caso_ok'] = predicciones
    df_resultado['Justificación'] = justificaciones

    # Estadísticas
    validos = [p for p in predicciones if p != -1]
    errores = [p for p in predicciones if p == -1]
    print(f'\nResultado ')
    print(f'  Total clasificadas: {len(validos)} de {len(df)}')
    print(f'  caso_ok = 1: {predicciones.count(1)}')
    print(f'  caso_ok = 0: {predicciones.count(0)}')
    if errores:
        print(f'  Errores:     {len(errores)} (revisar columna Justificación)')

    # Guardar
    output = f'resultado_{CLASIFICADOR_ACTIVO}.xlsx'
    df_resultado.to_excel(output, index=False)
    print(f'\n✅ Guardado: {output}')
    print('   → Evaluar con evaluacion.ipynb cambiando FILE_PREDICCIONES')

if __name__ == '__main__':
    main()
