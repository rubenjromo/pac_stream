import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Calculadora de Resistencia de Concreto con PAC",
    page_icon="🏗️",
    layout="wide"
)

# --- CARGAR MODELO Y ESCALADOR (se cachean para mayor eficiencia) ---
@st.cache_resource
def load_assets():
    """Carga el modelo y el escalador desde los archivos .pkl"""
    try:
        model = joblib.load('modelo_concreto_unificado.pkl')
        scaler = joblib.load('escalador_concreto_unificado.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: No se encontraron los archivos 'modelo_concreto_unificado.pkl' o 'escalador_concreto_unificado.pkl'.")
        st.error("Asegúrate de que los archivos estén en la misma carpeta que 'app.py' y que los nombres no contengan espacios.")
        return None, None

model, scaler = load_assets()

# Definir el orden correcto de las columnas para el modelo
FEATURE_COLUMNS = [
    'cemento_kg_m3', 'pac_kg_m3', 'escoria_alto_horno_kg_m3', 'ceniza_volante_kg_m3',
    'agua_kg_m3', 'superplastificante_kg_m3', 'agregado_grueso_kg_m3',
    'agregado_fino_kg_m3', 'activador_quimico_pct', 'edad_curado_dias'
]

# --- INTERFAZ DE USUARIO ---
st.title("🏗️ Calculadora Predictiva de Resistencia del Concreto")
st.markdown(
    "Esta herramienta, basada en la investigación de tesis, predice la evolución de la resistencia a compresión del concreto, "
    "evaluando el impacto del uso de lodo de PAC como material cementicio suplementario."
)
st.write("---")


# --- PANEL LATERAL PARA ENTRADA DE DATOS ---
st.sidebar.header("Parámetros de la Mezcla")
st.sidebar.markdown("Introduzca los valores de su mezcla por metro cúbico (m³).")

def user_inputs():
    """Crea los campos de entrada en la barra lateral y devuelve un diccionario con los valores."""
    inputs = {
        'cemento_kg_m3': st.sidebar.number_input("Cemento (kg/m³)", min_value=0.0, value=350.0, step=5.0),
        'pac_kg_m3': st.sidebar.number_input("Lodo de PAC (kg/m³)", min_value=0.0, value=50.0, step=5.0),
        'activador_quimico_pct': st.sidebar.number_input("Activador Químico (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.5),
        'escoria_alto_horno_kg_m3': st.sidebar.number_input("Escoria de Alto Horno (kg/m³)", min_value=0.0, value=0.0, step=5.0),
        'ceniza_volante_kg_m3': st.sidebar.number_input("Ceniza Volante (kg/m³)", min_value=0.0, value=0.0, step=5.0),
        'agua_kg_m3': st.sidebar.number_input("Agua (kg/m³)", min_value=0.0, value=170.0, step=2.0),
        'superplastificante_kg_m3': st.sidebar.number_input("Superplastificante (kg/m³)", min_value=0.0, value=0.0, step=0.5),
        'agregado_grueso_kg_m3': st.sidebar.number_input("Agregado Grueso (kg/m³)", min_value=0.0, value=1050.0, step=10.0),
        'agregado_fino_kg_m3': st.sidebar.number_input("Agregado Fino (kg/m³)", min_value=0.0, value=750.0, step=10.0),
    }
    return inputs

if model and scaler:
    input_data = user_inputs()
    
    # Botón principal para ejecutar el cálculo
    if st.sidebar.button("Calcular Evolución de Resistencia", type="primary"):
        
        # --- LÓGICA DE PREDICCIÓN (Adaptada de tu script) ---
        edades_curado = [7, 14, 28, 56, 90]

        # 1. Preparar datos para la mezcla del usuario
        mezcla_usuario_list = []
        for edad in edades_curado:
            fila = input_data.copy()
            fila['edad_curado_dias'] = edad
            mezcla_usuario_list.append(fila)
        df_usuario = pd.DataFrame(mezcla_usuario_list, columns=FEATURE_COLUMNS)

        # 2. Preparar datos para la mezcla de control (SIN PAC)
        inputs_sin_pac = input_data.copy()
        inputs_sin_pac['pac_kg_m3'] = 0
        inputs_sin_pac['activador_quimico_pct'] = 0

        mezcla_sin_pac_list = []
        for edad in edades_curado:
            fila_sin_pac = inputs_sin_pac.copy()
            fila_sin_pac['edad_curado_dias'] = edad
            mezcla_sin_pac_list.append(fila_sin_pac)
        df_sin_pac = pd.DataFrame(mezcla_sin_pac_list, columns=FEATURE_COLUMNS)

        # 3. Realizar predicciones
        predicciones_usuario = model.predict(scaler.transform(df_usuario)).clip(min=0)
        predicciones_sin_pac = model.predict(scaler.transform(df_sin_pac)).clip(min=0)
        
        # 4. Calcular la mejora
        epsilon = 1e-6
        mejora_pct = ((predicciones_usuario - predicciones_sin_pac) / (predicciones_sin_pac + epsilon)) * 100

        # --- MOSTRAR RESULTADOS ---
        st.subheader("Resultados de la Simulación")
        
        resultados_df = pd.DataFrame({
            'Edad de Curado (días)': edades_curado,
            'Resistencia Mezcla con PAC (MPa)': predicciones_usuario,
            'Resistencia Control sin PAC (MPa)': predicciones_sin_pac,
            'Variación (%)': mejora_pct
        })
        
        # Formatear y mostrar la tabla de resultados
        st.dataframe(
            resultados_df.style.format({
                'Resistencia Mezcla con PAC (MPa)': '{:.2f}',
                'Resistencia Control sin PAC (MPa)': '{:.2f}',
                'Variación (%)': '{:+.2f}%'
            }).apply(
                lambda x: ['background-color: #d4edda' if x.name == 'Variación (%)' and v > 0 else 'background-color: #f8d7da' if x.name == 'Variación (%)' and v < 0 else '' for v in x],
                axis=1
            ),
            use_container_width=True
        )

        st.info(
            "La columna 'Variación (%)' muestra la ganancia o pérdida de resistencia de su mezcla en comparación "
            "con una mezcla de control idéntica pero sin lodo de PAC ni activador químico."
        )

        # Mostrar gráfico de la evolución
        st.subheader("Evolución Comparativa de la Resistencia")
        chart_data = resultados_df.rename(columns={
            'Resistencia Mezcla con PAC (MPa)': 'Mezcla con PAC',
            'Resistencia Control sin PAC (MPa)': 'Control sin PAC'
        }).set_index('Edad de Curado (días)')[['Mezcla con PAC', 'Control sin PAC']]
        
        st.line_chart(chart_data)

    else:
        st.info("Ajuste los parámetros en el panel de la izquierda y presione 'Calcular' para ver los resultados.")
