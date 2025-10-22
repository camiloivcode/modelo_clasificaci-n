# generar_excel_prueba.py
import pandas as pd

try:
    # CREAR DATOS DE PRUEBA VARIADOS
    datos_prueba = {
        'Comentario': [
            # SEGURIDAD
            "las calles están muy oscuras y peligrosas por la noche",
            "hay robos frecuentes en el transporte público",
            "no hay policía en nuestro barrio",
            "asaltan a estudiantes cerca del colegio",
            "falta iluminación en el vecindario",
            
            # EDUCACIÓN
            "los niños no tienen escuela en esta comunidad",
            "faltan profesores de matemáticas",
            "no hay biblioteca para estudiar",
            "estudiantes sin acceso a internet",
            "aulas del colegio en mal estado",
            
            # MEDIO AMBIENTE
            "la basura no se recoge hace semanas",
            "el río está contaminado con desechos",
            "queman basura cerca de las casas",
            "mal olor por alcantarillas tapadas",
            "residuos acumulados en el parque",
            
            # SALUD
            "centro de salud sin médicos",
            "faltan medicinas para niños enfermos",
            "no hay ambulancia para emergencias",
            "hospital siempre lleno",
            "ancianos sin acceso a medicamentos",
            
            # CASOS MIXTOS
            "estudiantes se enferman por agua contaminada del colegio",
            "profesores no pueden llegar por inseguridad",
            "sin luz en hospital para emergencias",
            "estudiantes sin agua potable en colegio",
            "basura atrae ratas que enferman niños"
        ]
    }

    # CREAR DATAFRAME
    df_prueba = pd.DataFrame(datos_prueba)

    # GUARDAR COMO EXCEL
    nombre_archivo = "comentarios_prueba.xlsx"
    df_prueba.to_excel(nombre_archivo, index=False)

    print(f"✅ Excel de prueba creado: {nombre_archivo}")
    print(f"📊 Total de comentarios: {len(df_prueba)}")
    print("📋 Distribución:")
    print("- Seguridad: 5 comentarios")
    print("- Educación: 5 comentarios") 
    print("- Medio Ambiente: 5 comentarios")
    print("- Salud: 5 comentarios")
    print("- Casos Mixtos: 5 comentarios")
    print("\n🎯 ¡Sube este archivo a tu clasificador!")

except ImportError:
    print("❌ Error: Necesitas instalar openpyxl")
    print("💻 Ejecuta: pip install openpyxl")
except Exception as e:
    print(f"❌ Error: {e}")