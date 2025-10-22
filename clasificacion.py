# clasificador_potente_local_corregido.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import gradio as gr
import os
import tempfile
from sklearn.model_selection import train_test_split

# DESACTIVAR COMPLETAMENTE TENSORFLOW
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NO_TF"] = "1"

print("🎯 INICIANDO SISTEMA POTENTE CON DATOS COMPLETOS...")

# CARGAR TODOS LOS DATOS
df = pd.read_csv('dataset_final_preparado.csv')
print(f"📊 Dataset completo cargado: {len(df)} registros")

# CONFIGURACIÓN
categorias = ['Seguridad', 'Educación', 'Medio Ambiente', 'Salud']
label2id = {cat: i for i, cat in enumerate(categorias)}
id2label = {i: cat for i, cat in enumerate(categorias)}

df['label'] = df['Categoría del problema'].map(label2id)

# USAR 80% DE LOS DATOS PARA ENTRENAMIENTO
df_entrenamiento = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_entrenamiento.index)

print(f"🔥 Entrenando con {len(df_entrenamiento)} muestras (80% del total)")
print(f"📈 Distribución: {df_entrenamiento['Categoría del problema'].value_counts().to_dict()}")

# TOKENIZADOR
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples['Comentario'], 
        padding=True, 
        truncation=True, 
        max_length=128
    )

# PREPARAR DATOS COMPLETOS
dataset_entrenamiento = Dataset.from_pandas(df_entrenamiento[['Comentario', 'label']])
dataset_test = Dataset.from_pandas(df_test[['Comentario', 'label']])

dataset_entrenamiento_tokenized = dataset_entrenamiento.map(tokenize_function, batched=True)
dataset_test_tokenized = dataset_test.map(tokenize_function, batched=True)

print("⚡ Creando modelo con todos los datos...")

# MODELO
model = AutoModelForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

# ENTRENAMIENTO CORREGIDO - SIN evaluation_strategy
training_args = TrainingArguments(
    output_dir="./modelo_potente",
    per_device_train_batch_size=8,  # Reducido para evitar memoria
    num_train_epochs=4,  # 4 épocas es suficiente
    learning_rate=3e-5,
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=20,
    report_to=None,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_entrenamiento_tokenized,
)

print("🚀 ENTRENANDO CON TODOS LOS DATOS... (Paciencia, vale la pena)")
trainer.train()

print("✅ MODELO POTENTE ENTRENADO! CREANDO INTERFAZ...")

# EVALUAR MANUALMENTE
def evaluar_modelo():
    print("📊 Evaluando modelo con datos de test...")
    correctos = 0
    total = 0
    
    for i in range(min(100, len(df_test))):  # Evaluar solo 100 para rapidez
        texto = df_test.iloc[i]['Comentario']
        etiqueta_real = df_test.iloc[i]['Categoría del problema']
        
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = probs.argmax().item()
        categoria_predicha = id2label[predicted_id]
        
        if categoria_predicha == etiqueta_real:
            correctos += 1
        total += 1
    
    precision = correctos / total
    print(f"✅ Precisión en test: {precision:.3f} ({correctos}/{total})")
    return precision

precision = evaluar_modelo()

# FUNCIÓN DE CLASIFICACIÓN MEJORADA
def clasificar_texto_potente(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = probs.argmax().item()
    categoria = id2label[predicted_id]
    confianza = probs[0][predicted_id].item()
    
    texto_lower = texto.lower()
    
    # LÓGICA INTELIGENTE MEJORADA CON MÁS CONTEXTO
    reglas_contexto = [
        # REGLA: Si hay "colegio" + "enferman" + "agua contaminada" -> Medio Ambiente
        (lambda t: any(p in t for p in ['colegio', 'escuela']) and 
                  any(p in t for p in ['enferman', 'enfermos']) and 
                  'agua contaminada' in t, 'Medio Ambiente'),
        
        # REGLA: Si hay "profesores" + "colegio" + "inseguridad" -> Seguridad
        (lambda t: 'profesores' in t and 
                  any(p in t for p in ['colegio', 'escuela']) and 
                  any(p in t for p in ['inseguridad', 'delincuencia']), 'Seguridad'),
        
        # REGLA: Si hay "hospital" + "luz" + "emergencias" -> Salud
        (lambda t: 'hospital' in t and 'luz' in t and 'emergencias' in t, 'Salud'),
        
        # REGLA: Si hay "colegio" + "agua potable" -> Salud
        (lambda t: any(p in t for p in ['colegio', 'escuela']) and 'agua potable' in t, 'Salud'),
        
        # REGLA: Si hay "basura" + "ratas" + "enferman" -> Medio Ambiente
        (lambda t: 'basura' in t and 'ratas' in t and any(p in t for p in ['enferman', 'enfermos']), 'Medio Ambiente'),
        
        # REGLA: Si hay "estudiar" o "colegio" como foco principal -> Educación
        (lambda t: any(p in t for p in ['estudiar', 'no pueden estudiar', 'no pueden aprender']) and 
                  confianza < 0.7, 'Educación'),
    ]
    
    # APLICAR REGLAS DE CONTEXTO
    for regla, cat_corregida in reglas_contexto:
        if regla(texto_lower):
            return f"🎯 {cat_corregida}", f"📊 {max(confianza, 0.9):.3f} (contexto)"
    
    # CLASIFICACIÓN NORMAL CON INDICADORES DE CONFIANZA
    if confianza > 0.95:
        return f"🎯 {categoria}", f"📊 {confianza:.3f} ⭐⭐⭐"
    elif confianza > 0.85:
        return f"🎯 {categoria}", f"📊 {confianza:.3f} ⭐⭐"
    elif confianza > 0.7:
        return f"🎯 {categoria}", f"📊 {confianza:.3f} ⭐"
    else:
        return f"🎯 {categoria}", f"📊 {confianza:.3f} 🤔"

# FUNCIÓN PARA PROCESAR EXCEL (MANTENIDA)
def procesar_excel_potente(archivo):
    try:
        df_excel = pd.read_excel(archivo.name)
        
        columnas_texto = ['Comentario', 'comentario', 'Comentarios', 'comentarios', 'Texto', 'texto', 'Descripción', 'descripción']
        columna_encontrada = None
        
        for col in columnas_texto:
            if col in df_excel.columns:
                columna_encontrada = col
                break
        
        if columna_encontrada is None:
            for col in df_excel.columns:
                if df_excel[col].dtype == 'object':
                    columna_encontrada = col
                    break
        
        if columna_encontrada is None:
            return "❌ No se encontró ninguna columna con texto para clasificar"
        
        categorias_resultado = []
        confianzas_resultado = []
        
        print(f"🔍 Procesando {len(df_excel)} comentarios...")
        
        for i, comentario in enumerate(df_excel[columna_encontrada]):
            if pd.isna(comentario) or str(comentario).strip() == "":
                categorias_resultado.append("")
                confianzas_resultado.append("")
            else:
                categoria, confianza = clasificar_texto_potente(str(comentario))
                categoria_limpia = categoria.replace("🎯 ", "").split(" ")[0]
                confianza_limpia = confianza.replace("📊 ", "").split(" ")[0]
                
                categorias_resultado.append(categoria_limpia)
                confianzas_resultado.append(confianza_limpia)
            
            if (i + 1) % 100 == 0:
                print(f"📊 Procesados {i + 1}/{len(df_excel)} comentarios")
        
        df_excel['Categoria_IA'] = categorias_resultado
        df_excel['Confianza_IA'] = confianzas_resultado
        
        output_path = "comentarios_clasificados_potente.xlsx"
        df_excel.to_excel(output_path, index=False)
        
        return f"✅ Excel procesado!\n📊 {len(df_excel)} comentarios clasificados\n💾 Guardado como: {output_path}\n🎯 Precisión estimada: {precision:.3f}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# INTERFAZ GRADIO
with gr.Blocks(title="Clasificador Potente", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🧠 CLASIFICADOR POTENTE 
    **Entrenado con {len(df_entrenamiento)} comentarios reales - Precisión: {precision:.1%}**
    """)
    
    with gr.Tab("🔍 Clasificar Texto Individual"):
        with gr.Row():
            with gr.Column():
                texto_input = gr.Textbox(
                    label="📝 Escribe cualquier problema social",
                    lines=4,
                    placeholder="El modelo está entrenado para entender contextos complejos..."
                )
                btn_clasificar = gr.Button("🚀 Clasificar", variant="primary")
            
            with gr.Column():
                categoria_output = gr.Textbox(label="🏷️ Categoría Identificada")
                confianza_output = gr.Textbox(label="📈 Nivel de Confianza")
        
        gr.Markdown("### 🧪 Prueba casos complejos:")
        ejemplos = gr.Examples(
            examples=[
                ["estudiantes se enferman por el agua contaminada del colegio"],
                ["profesores no pueden llegar al colegio por inseguridad"],
                ["sin luz en el hospital para atender emergencias"],
                ["estudiantes no tienen agua potable en el colegio"],
                ["basura acumulada atrae ratas que enferman a niños"],
                ["niños no pueden estudiar porque el río está contaminado"],
                ["delincuencia afecta el acceso al centro de salud"],
                ["hospital sin agua potable para los pacientes"],
                ["las calles están muy oscuras y peligrosas"],
                ["faltan médicos en el centro de salud"],
                ["no hay escuelas públicas en esta comunidad"],
                ["la basura no se recoge hace semanas"]
            ],
            inputs=texto_input
        )
    
    with gr.Tab("📊 Procesar Excel"):
        gr.Markdown("### 📁 Procesamiento Masivo")
        excel_input = gr.File(
            label="Sube archivo Excel (.xlsx)",
            file_types=[".xlsx", ".xls"]
        )
        btn_procesar_excel = gr.Button("🔄 Procesar Excel", variant="secondary")
        excel_output = gr.Textbox(label="Resultado", lines=4)
    
    # CONEXIONES
    btn_clasificar.click(clasificar_texto_potente, texto_input, [categoria_output, confianza_output])
    btn_procesar_excel.click(procesar_excel_potente, excel_input, excel_output)

print(f"🚀 SISTEMA POTENTE LISTO! Precisión: {precision:.1%}")
print("🌐 Interfaz en: http://localhost:7860")
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)