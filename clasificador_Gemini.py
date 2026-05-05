"""
Proyecto: Derecho a las Víctimas
Script: Prueba de Gemini via API de Google

Descripción:
Este script clasifica normas legales argentinas según si consagran o no derechos de las víctimas, 
utilizando el modelo Gemini de Google a través de su API. 

Input:
    - test.xlsx
    - resultado_gemini_parcial.xlsx 

Output:
    - resultado_gemini_parte3.xlsx (normas 42-62)
    - resultado_gemini_final.xlsx (todas las normas mergeadas)
"""

import pandas as pd
import os
import time
from dotenv import load_dotenv
load_dotenv()

# Configuración
FILE_TEST             = 'test.xlsx'
FILE_PARCIAL          = 'resultado_gemini_parcial.xlsx'
FILE_PARTE3           = 'resultado_gemini_parte3.xlsx'
FILE_FINAL            = 'resultado_gemini_final.xlsx'
CLASIFICADOR          = 'gemini'
GOOGLE_API_KEY        = os.getenv('GOOGLE_API_KEY', '')
MODELO                = {'gemini': 'gemini-2.5-flash'}
PAUSA_ENTRE_LLAMADAS  = 15  # segundos — no bajar para no agotar el cupo

# Prompt
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
Respondé con el número 1 o 0 seguido de dos puntos y una oración breve explicando tu decisión. Ejemplo: '1: La norma establece asistencia directa a víctimas de inundación.'"""


def construir_prompt_usuario(row):
    titulo    = str(row.get('Título', '')).strip()   if pd.notna(row.get('Título'))    else ''
    resumen   = str(row.get('Resumen', '')).strip()  if pd.notna(row.get('Resumen'))   else ''
    articulos = str(row.get('Artículos', '')).strip() if pd.notna(row.get('Artículos')) else ''

    texto = f"Tipo: {row.get('Tipo', '')}\nNúmero: {row.get('Número', '')}\nTítulo: {titulo}"
    if resumen:
        texto += f"\nResumen: {resumen}"
    if articulos:
        texto += f"\nArtículos relevantes: {articulos[:500]}"
    texto += "\n\n¿Esta norma consagra derechos de las víctimas?"
    return texto


def parsear_respuesta(respuesta):
    respuesta = str(respuesta).strip()
    if respuesta.startswith('1'):
        return 1, respuesta
    elif respuesta.startswith('0'):
        return 0, respuesta
    else:
        return -1, f"Respuesta inválida: {respuesta}"
    

def clasificar_con_gemini(df):
    try:
        from google import genai
    except ImportError:
        os.system('pip install google-genai -q')
        from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    predicciones   = []
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

def main():
    print('=' * 60)
    print(f'  Clasificador: GEMINI — Parte 3 (normas 42 en adelante)')
    print('=' * 60)

    # Cargar y filtrar — solo las normas pendientes
    df = pd.read_excel(FILE_TEST)
    df['Número'] = df['Número'].astype(str).str.strip()
    df = df.iloc[41:]  # arranca desde la norma 42
    print(f'\nNormas a clasificar: {len(df)}')

    # Clasificar
    print(f'\nClasificando con Gemini...')
    predicciones, justificaciones = clasificar_con_gemini(df)

    # Armar resultado parte 3
    df_parte3 = df[['Tipo', 'Número', 'Título']].copy()
    df_parte3['caso_ok']      = predicciones
    df_parte3['Justificación'] = justificaciones

    # Estadísticas
    validos = [p for p in predicciones if p != -1]
    errores = [p for p in predicciones if p == -1]
    print(f'  Total clasificadas: {len(validos)} de {len(df)}')
    print(f'  caso_ok = 1: {predicciones.count(1)}')
    print(f'  caso_ok = 0: {predicciones.count(0)}')
    if errores:
        print(f'  Errores:     {len(errores)} (revisar columna Justificación)')

    # Guardar parte 3
    df_parte3.to_excel(FILE_PARTE3, index=False)
    print(f'\n✅ Guardado: {FILE_PARTE3}')

    # Mergear con el parcial anterior
    print('\nMergeando con resultado parcial anterior...')
    df_parcial = pd.read_excel(FILE_PARCIAL)
    df_final   = pd.concat([df_parcial, df_parte3]).reset_index(drop=True)
    df_final.to_excel(FILE_FINAL, index=False)
    print(f'✅ Guardado: {FILE_FINAL}')
    print(f'\nTotal final: {len(df_final)} normas')
    print(f'  caso_ok = 1: {(df_final["caso_ok"]==1).sum()}')
    print(f'  caso_ok = 0: {(df_final["caso_ok"]==0).sum()}')
    if (df_final['caso_ok']==-1).sum() > 0:
        print(f'  Errores: {(df_final["caso_ok"]==-1).sum()} — necesitás una parte 4')


if __name__ == '__main__':
    main()
