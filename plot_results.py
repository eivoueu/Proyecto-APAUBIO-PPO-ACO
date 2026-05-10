import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def cargar_datos_experimento(nombre_mapa, id_experimento, semillas):
    datos = []
    for semilla in semillas:
        ruta = f"data/resultados_{nombre_mapa}_{id_experimento}_semilla_{semilla}.csv"
        if os.path.exists(ruta):
            df = pd.read_csv(ruta, engine='python')
            datos.append(df)
    return datos

def calcular_metricas(datos_df_list, ventana=100):
    if not datos_df_list:
        return None
    
    recompensas = np.array([df['recompensa'].values for df in datos_df_list])
    media = np.mean(recompensas, axis=0)
    std = np.std(recompensas, axis=0)
    
    media_suavizada = pd.Series(media).rolling(window=ventana, min_periods=1).mean().values
    std_suavizada = pd.Series(std).rolling(window=ventana, min_periods=1).mean().values
    
    idx_10_porciento = int(len(media) * 0.9)
    convergencia_por_semilla = [np.mean(r[idx_10_porciento:]) for r in recompensas]
    convergencia_media = np.mean(convergencia_por_semilla)
    inestabilidad = np.std(media[idx_10_porciento:])
    
    return {
        'media_suavizada': media_suavizada, 
        'std_suavizada': std_suavizada, 
        'convergencia_semillas': convergencia_por_semilla,
        'convergencia_media': convergencia_media, 
        'inestabilidad': inestabilidad,
        'df_base': datos_df_list[0]
    }

def add_labels(ax, rects, formato='{:.2f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(formato.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

def plotear_resultados():
    os.makedirs("figures", exist_ok=True)
    semillas = [42, 101, 202, 303, 404]
    mapa = "Normal"
    ventana = 200

    print("\n" + "="*60)
    print("ANALIZANDO REGISTROS EXPERIMENTALES Y GENERANDO GRÁFICAS")
    print("="*60)

    ids_experimentos = [
        "Ablacion_SoloPPO", "Hibrido_Base", "Hibrido_LRAlto", 
        "Hibrido_EvapAlta", "Hibrido_RedGrande", 
        "Hibrido_AlphaBajo", "Hibrido_AlphaBajo_EvapAlta"
    ]
    labels_experimentos = [
        "Solo PPO", "Híbrido Base", "LR Alto", 
        "Evaporación Alta", "Red Grande", 
        "Alpha Bajo", "Alpha Bajo + Evap Alta"
    ]
    colores_experimentos = ['red', 'blue', 'orange', 'green', 'purple', 'cyan', 'magenta']
    
    metricas_todas = []
    for id_exp in ids_experimentos:
        datos = cargar_datos_experimento(mapa, id_exp, semillas)
        metricas_todas.append(calcular_metricas(datos, ventana))

    m_ablacion = metricas_todas[0]
    m_base = metricas_todas[1]

    if not m_base or not m_ablacion:
        print("Error crítico: No se encontraron datos en 'data/'.")
        return

    episodios = m_base['df_base']['episodio'].values


    # GRÁFICA 1: ANÁLISIS DE APRENDIZAJE Y ABLACIÓN

    d_base = cargar_datos_experimento(mapa, "Hibrido_Base", semillas)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    colores_s = ["#FF5733", "#33FF57", "#3357FF", "#F333FF", "#FF33A1"]

    for i, df_s in enumerate(d_base):
        ax = axes[i]
        suavizada = df_s['recompensa'].rolling(window=ventana, min_periods=1).mean()
        ax.plot(df_s['episodio'], suavizada, color=colores_s[i], linewidth=1.5)
        ax.set_title(f'Semilla {semillas[i]} (Híbrido)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_ylabel('Recompensa')

    ax_total = axes[5]
    ax_total.plot(episodios, m_base['media_suavizada'], label='Media Híbrido', color='blue', linewidth=2)
    ax_total.fill_between(episodios, 
                          m_base['media_suavizada'] - m_base['std_suavizada'],
                          m_base['media_suavizada'] + m_base['std_suavizada'], 
                          color='blue', alpha=0.15)
    
    ax_total.plot(episodios, m_ablacion['media_suavizada'], label='Media Solo PPO', color='red', linewidth=1.5, linestyle='--')
    ax_total.fill_between(episodios, 
                          m_ablacion['media_suavizada'] - m_ablacion['std_suavizada'],
                          m_ablacion['media_suavizada'] + m_ablacion['std_suavizada'], 
                          color='red', alpha=0.1)
    
    ax_total.set_title('Comparativa Global y Estudio de Ablación', fontsize=10, fontweight='bold', color='darkred')
    ax_total.legend(fontsize='small', loc='lower right')
    ax_total.grid(True, alpha=0.3)

    plt.suptitle('Rendimiento de Aprendizaje: Desglose por Semillas', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figures/01_analisis_aprendizaje_detallado.png", dpi=300)
    plt.close()

    # GRÁFICA 2: DIAGNÓSTICO DEL COMPONENTE BIOINSPIRADO (feromonas y alpha)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_fero = 'tab:green'
    ax1.set_xlabel('Episodios')
    ax1.set_ylabel('Intensidad de Feromonas (ACO)', color=color_fero, fontweight='bold')
    
    df_diag = d_base[0] 
    line1 = ax1.plot(df_diag['episodio'], df_diag['feromona_max'].rolling(100).mean(), 
                     label='Feromona Máxima', color='darkgreen', linewidth=2)
    line2 = ax1.plot(df_diag['episodio'], df_diag['feromona_media'].rolling(100).mean(), 
                     label='Feromona Media', color='lightgreen', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_fero)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()  
    color_alpha = 'tab:purple'
    ax2.set_ylabel('Peso de Acoplamiento (Alpha)', color=color_alpha, fontweight='bold')  
    line3 = ax2.plot(df_diag['episodio'], df_diag['peso_aco_aplicado'], 
                     label='Peso Alpha (Decay)', color=color_alpha, linestyle=':', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_alpha)

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', frameon=True, shadow=True)

    plt.title('Dinámica Adaptativa Interna')
    fig.tight_layout()  
    plt.savefig("figures/02_diagnostico_bioinspirado.png", dpi=300)
    plt.close()

    # GRÁFICA 3: COMPARAR HIPERPARÁMETROS DEL HÍBRIDO

    labels_barrido = ["Base", "LR Alto", "Solo PPO", "Red Grande"]
    colores_b = ['blue', 'orange', 'red', 'purple']
    
    indices_g3 = [1, 2, 0, 4]
    vals_conv = [metricas_todas[i]['convergencia_media'] if metricas_todas[i] else 0 for i in indices_g3]
    vals_inest = [metricas_todas[i]['inestabilidad'] if metricas_todas[i] else 0 for i in indices_g3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    rects1 = ax1.bar(labels_barrido, vals_conv, color=colores_b, alpha=0.7)
    add_labels(ax1, rects1)
    ax1.set_title('Puntuación de Convergencia ', fontweight='bold')
    ax1.set_ylabel('Recompensa Media (Último 10%)')
    ax1.grid(axis='y', alpha=0.3)

    rects2 = ax2.bar(labels_barrido, vals_inest, color=colores_b, alpha=0.7)
    add_labels(ax2, rects2)
    ax2.set_title('Inestabilidad en Convergencia', fontweight='bold')
    ax2.set_ylabel('Desviación Estándar')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Comparativa Hiperparámetros', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figures/03_comparativa_hiperparametros.png", dpi=300)
    plt.close()

    # GRÁFICA 4: COMPARATIVA CONFIHURACIONES (Curvas de Aprendizaje)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex=True)
    axes = axes.flatten()
    axes[7].set_visible(False)

    for i in range(7):
        ax = axes[i]
        m_config = metricas_todas[i]
        
        ax.plot(episodios, m_base['media_suavizada'], color='gray', alpha=0.4, linewidth=2, label='Referencia (Base)')
        
        if m_config:
            ax.plot(episodios, m_config['media_suavizada'], color=colores_experimentos[i], linewidth=2, label=labels_experimentos[i])
            ax.fill_between(episodios, 
                            m_config['media_suavizada'] - m_config['std_suavizada'],
                            m_config['media_suavizada'] + m_config['std_suavizada'], 
                            color=colores_experimentos[i], alpha=0.15)
        
        ax.set_title(labels_experimentos[i], fontweight='bold')
        ax.set_ylabel('Recompensa')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize='small')

    axes[5].set_xlabel('Episodios')
    axes[6].set_xlabel('Episodios')

    plt.suptitle('Estudio Completo de Sensibilidad: Todas las Configuraciones', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("figures/04_comparativa_todas_curvas.png", dpi=300)
    plt.close()

    # GRÁFICA 5: MAPA DE CALOR (Alpha vs Evaporación: alto/bajo)

    print("Generando Mapa de Calor Bioinspirado...")
    try:
        m_ab = metricas_todas[5]
        m_abea = metricas_todas[6]
        m_ea = metricas_todas[3]

        matriz_calor = np.array([
            [m_ab['convergencia_media'] if m_ab else 0, m_abea['convergencia_media'] if m_abea else 0],
            [m_base['convergencia_media'] if m_base else 0, m_ea['convergencia_media'] if m_ea else 0]
        ])

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(matriz_calor, cmap='viridis', aspect='auto')
        
        ax.set_xticks([0, 1], labels=['Evap Baja (0.01)', 'Evap Alta (0.05)'])
        ax.set_yticks([0, 1], labels=['Alpha Bajo (0.5)', 'Alpha Alto (1.0)'])
        
        for i in range(2):
            for j in range(2):
                valor = matriz_calor[i, j]
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="white", fontweight='bold')

        ax.set_title("Sensibilidad Bioinspirada: Convergencia")
        fig.colorbar(cax, label='Recompensa Media')
        plt.tight_layout()
        plt.savefig("figures/05_mapa_calor_bioinspirado.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"No se pudo generar el mapa de calor: {e}")


    # GRÁFICA 6: COMPARATIVA DE TIEMPOS DE EJECUCIÓN 

    ruta_tiempos = "data/tiempos_ejecucion.csv"
    if os.path.exists(ruta_tiempos):
        df_tiempos = pd.read_csv(ruta_tiempos)
        df_tiempos['Etiqueta'] = df_tiempos['ID'].map(dict(zip(ids_experimentos, labels_experimentos)))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects_t = ax.bar(df_tiempos['Etiqueta'], df_tiempos['Tiempo_Minutos'], color=colores_experimentos, alpha=0.8)
        add_labels(ax, rects_t, formato='{:.1f}m')
        
        ax.set_title('Coste Computacional: Tiempo de Entrenamiento por Configuración', fontsize=14, fontweight='bold')
        ax.set_ylabel('Tiempo Total (Minutos)')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig("figures/06_comparativa_tiempos.png", dpi=300)
        plt.close()
        print(">>> Gráfica de tiempos de ejecución generada correctamente.")
    else:
        print("\n[Aviso] No se encontró 'data/tiempos_ejecucion.csv'. Debes ejecutar la Opción 2 del main para generarlo.")


    # CÁLCULO DE MÉTRICAS ESTADÍSTICAS

    print("\n" + "="*60)
    print("MÉTRICAS ESTÁNDAR Y RESULTADOS:")
    print("="*60)
    
    r_hibrido = m_base['convergencia_media']
    r_base = m_ablacion['convergencia_media']
    print(f"- Convergencia Media Híbrido (Base): {r_hibrido:.2f}")
    print(f"- Convergencia Media Línea Base (Solo PPO): {r_base:.2f}")
    
    delta = (r_hibrido - r_base) / abs(r_base) if r_base != 0 else 0
    print(f"- Mejora Relativa del Híbrido (Delta): {delta * 100:.2f}%")
    print(f"- Robustez Híbrido (Inestabilidad final): {m_base['inestabilidad']:.4f}")
    
    max_retorno = max(np.max(m_base['media_suavizada']), np.max(m_ablacion['media_suavizada']))
    objetivo_80 = max_retorno * 0.8
    
    idx_80_h = np.argmax(m_base['media_suavizada'] >= objetivo_80)
    idx_80_a = np.argmax(m_ablacion['media_suavizada'] >= objetivo_80)
    
    print(f"- Eficiencia Híbrido: Alcanza 80% ({objetivo_80:.2f}) en episodio {episodios[idx_80_h]}")
    print(f"- Eficiencia Solo PPO: Alcanza 80% en episodio {episodios[idx_80_a]}")

    stat, p_value = stats.ttest_ind(m_base['convergencia_semillas'], 
                                    m_ablacion['convergencia_semillas'], 
                                    equal_var=False)
    
    print(f"- T-Test Estadística (t de Welch): p-value = {p_value:.6f}")
    if os.path.exists(ruta_tiempos):
        print("\n[TIEMPOS DE EJECUCIÓN (Minutos)]")
        tiempo_total = 0
        for index, row in df_tiempos.iterrows():
            print(f"- {row['Etiqueta']}: {row['Tiempo_Minutos']:.2f} min")
            tiempo_total += row['Tiempo_Minutos']
        print("-" * 35)
        print(f"- Tiempo Total: {tiempo_total:.2f} min")
        
    print("\n>>> Set completo de gráficas profesionales exportado en 'figures/' <<<")

if __name__ == "__main__":
    plotear_resultados()