import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import io
import base64
from datetime import datetime
import os

from utils.excel_utils import UsersExcelHandler, PredictionsExcelHandler

class VisualizationService:
    """Servicio para generar visualizaciones de datos"""
    
    def __init__(self):
        self.users_handler = UsersExcelHandler()
        self.predictions_handler = PredictionsExcelHandler()
        
        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configurar fuentes y colores
        self.colors = {
            'primary': '#3498db',
            'secondary': '#e74c3c',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'dark': '#34495e'
        }
    
    def create_plot_base64(self, fig) -> str:
        """Convertir figura matplotlib a base64"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
    
    def generate_age_distribution(self) -> Dict[str, Any]:
        """Generar gráfico de distribución de edades"""
        try:
            df = self.users_handler.read_users()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histograma
            ax.hist(df['age'], bins=15, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Edad')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Edades de Usuarios')
            ax.grid(True, alpha=0.3)
            
            # Estadísticas
            mean_age = df['age'].mean()
            median_age = df['age'].median()
            
            ax.axvline(mean_age, color=self.colors['secondary'], linestyle='--', label=f'Media: {mean_age:.1f}')
            ax.axvline(median_age, color=self.colors['success'], linestyle='--', label=f'Mediana: {median_age:.1f}')
            ax.legend()
            
            image_base64 = self.create_plot_base64(fig)
            
            return {
                'title': 'Distribución de Edades',
                'image': image_base64,
                'statistics': {
                    'mean': float(mean_age),
                    'median': float(median_age),
                    'std': float(df['age'].std()),
                    'min': int(df['age'].min()),
                    'max': int(df['age'].max())
                }
            }
            
        except Exception as e:
            return {'error': f'Error generando distribución de edades: {str(e)}'}
    
    def generate_social_media_vs_performance(self) -> Dict[str, Any]:
        """Generar gráfico de redes sociales vs rendimiento académico"""
        try:
            df = self.users_handler.read_users()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot
            scatter = ax.scatter(df['social_media_usage'], df['academic_performance'], 
                               alpha=0.6, c=df['age'], cmap='viridis', s=60)
            
            # Línea de tendencia
            z = np.polyfit(df['social_media_usage'], df['academic_performance'], 1)
            p = np.poly1d(z)
            ax.plot(df['social_media_usage'], p(df['social_media_usage']), 
                   color=self.colors['secondary'], linestyle='--', linewidth=2)
            
            ax.set_xlabel('Uso de Redes Sociales (horas/día)')
            ax.set_ylabel('Rendimiento Académico (%)')
            ax.set_title('Relación entre Uso de Redes Sociales y Rendimiento Académico')
            ax.grid(True, alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Edad')
            
            # Correlación
            correlation = df['social_media_usage'].corr(df['academic_performance'])
            ax.text(0.05, 0.95, f'Correlación: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            image_base64 = self.create_plot_base64(fig)
            
            return {
                'title': 'Redes Sociales vs Rendimiento Académico',
                'image': image_base64,
                'correlation': float(correlation),
                'insights': [
                    f'Correlación {"negativa" if correlation < 0 else "positiva"} de {abs(correlation):.3f}',
                    f'{"Fuerte" if abs(correlation) > 0.5 else "Moderada" if abs(correlation) > 0.3 else "Débil"} relación',
                    f'Tendencia {"descendente" if correlation < 0 else "ascendente"}'
                ]
            }
            
        except Exception as e:
            return {'error': f'Error generando gráfico de correlación: {str(e)}'}
    
    def generate_platform_distribution(self) -> Dict[str, Any]:
        """Generar gráfico de distribución de plataformas"""
        try:
            df = self.users_handler.read_users()
            
            platform_counts = df['main_platform'].value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de barras
            colors = plt.cm.Set3(np.linspace(0, 1, len(platform_counts)))
            bars = ax1.bar(platform_counts.index, platform_counts.values, color=colors)
            ax1.set_xlabel('Plataforma Principal')
            ax1.set_ylabel('Número de Usuarios')
            ax1.set_title('Distribución de Plataformas Principales')
            ax1.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Gráfico de pastel
            ax2.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax2.set_title('Distribución Porcentual de Plataformas')
            
            plt.tight_layout()
            image_base64 = self.create_plot_base64(fig)
            
            return {
                'title': 'Distribución de Plataformas Principales',
                'image': image_base64,
                'data': platform_counts.to_dict(),
                'most_popular': platform_counts.index[0],
                'least_popular': platform_counts.index[-1]
            }
            
        except Exception as e:
            return {'error': f'Error generando distribución de plataformas: {str(e)}'}
    
    def generate_performance_by_gender(self) -> Dict[str, Any]:
        """Generar gráfico de rendimiento por género"""
        try:
            df = self.users_handler.read_users()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Box plot
            box_plot = df.boxplot(column='academic_performance', by='gender', ax=ax, 
                                 patch_artist=True, return_type='dict')
            
            # Personalizar colores
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]
            for patch, color in zip(box_plot['academic_performance']['boxes'], colors[:len(box_plot['academic_performance']['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Género')
            ax.set_ylabel('Rendimiento Académico (%)')
            ax.set_title('Distribución de Rendimiento Académico por Género')
            plt.suptitle('')  # Remove automatic title
            
            # Estadísticas por género
            gender_stats = df.groupby('gender')['academic_performance'].agg(['mean', 'median', 'std']).round(2)
            
            image_base64 = self.create_plot_base64(fig)
            
            return {
                'title': 'Rendimiento Académico por Género',
                'image': image_base64,
                'statistics': gender_stats.to_dict(),
                'insights': [
                    f'Promedio general: {df["academic_performance"].mean():.2f}%',
                    f'Mejor rendimiento promedio: {gender_stats["mean"].idxmax()}',
                    f'Mayor variabilidad: {gender_stats["std"].idxmax()}'
                ]
            }
            
        except Exception as e:
            return {'error': f'Error generando gráfico por género: {str(e)}'}
    
    def generate_correlation_heatmap(self) -> Dict[str, Any]:
        """Generar mapa de calor de correlaciones"""
        try:
            df = self.users_handler.read_users()
            
            # Seleccionar columnas numéricas
            numeric_cols = ['age', 'social_media_usage', 'academic_performance', 'study_hours']
            correlation_data = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Mapa de calor
            mask = np.triu(np.ones_like(correlation_data, dtype=bool))
            sns.heatmap(correlation_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, linewidths=0.5, ax=ax, fmt='.3f')
            
            ax.set_title('Matriz de Correlación entre Variables')
            plt.tight_layout()
            
            image_base64 = self.create_plot_base64(fig)
            
            # Encontrar correlaciones más fuertes
            corr_values = []
            for i in range(len(correlation_data.columns)):
                for j in range(i+1, len(correlation_data.columns)):
                    corr_values.append({
                        'variables': f"{correlation_data.columns[i]} - {correlation_data.columns[j]}",
                        'correlation': correlation_data.iloc[i, j]
                    })
            
            strongest_positive = max(corr_values, key=lambda x: x['correlation'] if x['correlation'] > 0 else -1)
            strongest_negative = min(corr_values, key=lambda x: x['correlation'] if x['correlation'] < 0 else 1)
            
            return {
                'title': 'Matriz de Correlación',
                'image': image_base64,
                'correlation_matrix': correlation_data.to_dict(),
                'strongest_positive': strongest_positive,
                'strongest_negative': strongest_negative
            }
            
        except Exception as e:
            return {'error': f'Error generando mapa de correlación: {str(e)}'}
    
    def generate_study_hours_analysis(self) -> Dict[str, Any]:
        """Generar análisis de horas de estudio"""
        try:
            df = self.users_handler.read_users()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Distribución de horas de estudio
            ax1.hist(df['study_hours'], bins=15, color=self.colors['info'], alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Horas de Estudio por Semana')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('Distribución de Horas de Estudio')
            ax1.grid(True, alpha=0.3)
            
            # 2. Horas de estudio vs rendimiento
            ax2.scatter(df['study_hours'], df['academic_performance'], 
                       alpha=0.6, color=self.colors['primary'])
            z = np.polyfit(df['study_hours'], df['academic_performance'], 1)
            p = np.poly1d(z)
            ax2.plot(df['study_hours'], p(df['study_hours']), 
                    color=self.colors['secondary'], linestyle='--')
            ax2.set_xlabel('Horas de Estudio por Semana')
            ax2.set_ylabel('Rendimiento Académico (%)')
            ax2.set_title('Horas de Estudio vs Rendimiento')
            ax2.grid(True, alpha=0.3)
            
            # 3. Horas de estudio por nivel educativo
            education_study = df.groupby('education_level')['study_hours'].mean()
            bars = ax3.bar(education_study.index, education_study.values, 
                          color=[self.colors['success'], self.colors['warning'], self.colors['dark']])
            ax3.set_xlabel('Nivel Educativo')
            ax3.set_ylabel('Promedio de Horas de Estudio')
            ax3.set_title('Horas de Estudio por Nivel Educativo')
            ax3.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            # 4. Relación entre redes sociales y horas de estudio
            ax4.scatter(df['social_media_usage'], df['study_hours'], 
                       alpha=0.6, color=self.colors['warning'])
            z = np.polyfit(df['social_media_usage'], df['study_hours'], 1)
            p = np.poly1d(z)
            ax4.plot(df['social_media_usage'], p(df['social_media_usage']), 
                    color=self.colors['secondary'], linestyle='--')
            ax4.set_xlabel('Uso de Redes Sociales (horas/día)')
            ax4.set_ylabel('Horas de Estudio por Semana')
            ax4.set_title('Redes Sociales vs Horas de Estudio')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            image_base64 = self.create_plot_base64(fig)
            
            # Estadísticas
            study_performance_corr = df['study_hours'].corr(df['academic_performance'])
            social_study_corr = df['social_media_usage'].corr(df['study_hours'])
            
            return {
                'title': 'Análisis de Horas de Estudio',
                'image': image_base64,
                'statistics': {
                    'mean_study_hours': float(df['study_hours'].mean()),
                    'median_study_hours': float(df['study_hours'].median()),
                    'study_performance_correlation': float(study_performance_corr),
                    'social_study_correlation': float(social_study_corr),
                    'by_education_level': education_study.to_dict()
                },
                'insights': [
                    f'Correlación estudio-rendimiento: {study_performance_corr:.3f}',
                    f'Correlación redes sociales-estudio: {social_study_corr:.3f}',
                    f'Promedio de horas de estudio: {df["study_hours"].mean():.1f} horas/semana'
                ]
            }
            
        except Exception as e:
            return {'error': f'Error generando análisis de horas de estudio: {str(e)}'}
    
    def generate_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Generar dashboard completo con múltiples visualizaciones"""
        try:
            dashboard = {
                'title': 'Dashboard Completo de Análisis',
                'timestamp': datetime.now().isoformat(),
                'charts': {},
                'summary': {}
            }
            
            # Generar todas las visualizaciones
            dashboard['charts']['age_distribution'] = self.generate_age_distribution()
            dashboard['charts']['social_vs_performance'] = self.generate_social_media_vs_performance()
            dashboard['charts']['platform_distribution'] = self.generate_platform_distribution()
            dashboard['charts']['performance_by_gender'] = self.generate_performance_by_gender()
            dashboard['charts']['correlation_heatmap'] = self.generate_correlation_heatmap()
            dashboard['charts']['study_hours_analysis'] = self.generate_study_hours_analysis()
            
            # Resumen general
            df = self.users_handler.read_users()
            dashboard['summary'] = {
                'total_users': len(df),
                'average_age': float(df['age'].mean()),
                'average_social_media_usage': float(df['social_media_usage'].mean()),
                'average_academic_performance': float(df['academic_performance'].mean()),
                'average_study_hours': float(df['study_hours'].mean()),
                'main_insights': [
                    f"Usuario promedio: {df['age'].mean():.0f} años",
                    f"Uso promedio de redes sociales: {df['social_media_usage'].mean():.1f} horas/día",
                    f"Rendimiento promedio: {df['academic_performance'].mean():.1f}%",
                    f"Estudio promedio: {df['study_hours'].mean():.1f} horas/semana"
                ]
            }
            
            return dashboard
            
        except Exception as e:
            return {'error': f'Error generando dashboard completo: {str(e)}'}
    
    def save_plots_to_static(self) -> Dict[str, Any]:
        """Guardar plots en directorio static"""
        try:
            static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'plots')
            os.makedirs(static_dir, exist_ok=True)
            
            saved_files = {}
            
            # Generar y guardar cada plot
            plots = {
                'age_histogram': self.generate_age_distribution,
                'social_media_scatter': self.generate_social_media_vs_performance,
                'platform_distribution': self.generate_platform_distribution,
                'performance_by_gender': self.generate_performance_by_gender,
                'correlation_heatmap': self.generate_correlation_heatmap,
                'study_hours_analysis': self.generate_study_hours_analysis
            }
            
            for plot_name, plot_function in plots.items():
                try:
                    result = plot_function()
                    if 'image' in result:
                        # Decodificar base64 y guardar
                        image_data = base64.b64decode(result['image'])
                        file_path = os.path.join(static_dir, f'{plot_name}.png')
                        
                        with open(file_path, 'wb') as f:
                            f.write(image_data)
                        
                        saved_files[plot_name] = file_path
                        
                except Exception as e:
                    saved_files[plot_name] = f'Error: {str(e)}'
            
            return {
                'success': True,
                'saved_files': saved_files,
                'static_directory': static_dir
            }
            
        except Exception as e:
            return {'error': f'Error guardando plots: {str(e)}'}
