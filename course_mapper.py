import pandas as pd
import json

class CourseNameMapper:
    def __init__(self):
        # Master mapping of course names and their variations
        self.course_mapping = {
            # French Courses
            "Introduction à DHIS2": [
                "Introduction a DHIS2",
                "Introduction à DHIS2",
                "Introduction à  DHIS2",
                "Introduction Ã  DHIS2",
                "TEST Introduction Ã  DHIS2",
                "Introduction a  DHIS2"
            ],
            "Les principes fondamentaux de DHIS2": [
                "Les principes fondamentaux de DHIS2",
                "Les principes fondamentaux de  DHIS2",
                "Les principes fondamentaux de DHIS 2"
            ],
            "Les fondamentaux de DHIS2 événements": [
                "Les fondamentaux de DHIS2 evenements",
                "Les fondamentaux de DHIS2 événements",
                "Les fondamentaux de DHIS2 ÃvÃ©nements",
                "Les fondamentaux de DHIS2 Événements",
                "Les fondamentaux de DHIS2 ÃƒÂ©vÃƒÂ©nements",
                "Les fondamentaux de DHIS2 évènements"
            ],
            "Le paramétrage de DHIS2 agrégé": [
                "Le parametrage de DHIS2 agrege",
                "Le paramétrage de DHIS2 agrégé",
                "Le paramÃ©trage de DHIS2 agrÃ©gÃ©",
                "Le paramétrage de DHIS2 agrégée"
            ],
            "Les fondamentaux de la saisie et de la validation des données agrégées": [
                "Les fondamentaux de la saisie et de la validation des données agrégées",
                "Les fondamentaux de la saisie et de la validation des donnees agregees",
                "Les fondamentaux de la saisie et de la validation des données agrégées",
                "Les fondamentaux de la saisie et de la validation des donnÃ©es agrÃ©gÃ©es"
            ],

            # Spanish Courses
            "Introducción a DHIS2": [
                "Introduccion a DHIS2",
                "Introducción a DHIS2",
                "IntroducciÃ³n a DHIS2"
            ],
            "Fundamentos de Análisis de Datos Agregados en DHIS2": [
                "Fundamentos de Analisis de Datos Agregados en DHIS2",
                "Fundamentos de Análisis de Datos Agregados en DHIS2",
                "Fundamentos de AnÃ¡lisis de Datos Agregados en DHIS2"
            ],
            "Fundamentos de Configuración de Datos Agregados": [
                "Fundamentos de Configuracion de Datos Agregados",
                "Fundamentos de Configuración de Datos Agregados",
                "Fundamentos de ConfiguraciÃ³n de Datos Agregados"
            ],
            "Fundamentos de Captura y Validación de Datos Agregados": [
                "Fundamentos de Captura y Validación de Datos Agregados",
                "Fundamentos de Captura y ValidaciÃ³n de Datos Agregados"
            ],

            # English Courses
            "Introduction to DHIS2": [
                "Introduction to DHIS2",
                "DHIS2 Fundamentals",
                "DHIS2 Fundamentals (old)"
            ],
            "DHIS2 Events Fundamentals": ["DHIS2 Events Fundamentals"],
            "Aggregate Data Analysis Fundamentals": ["Aggregate Data Analysis Fundamentals"],
            "Aggregate Data Capture and Validation Fundamentals": ["Aggregate Data Capture and Validation Fundamentals"],
            "Aggregate Customization Fundamentals": ["Aggregate Customization Fundamentals"],
            "Data Quality Level 2 Academy": [
                "Data Quality Level 2 Academy",
                "Data Quality Level 2 - Online Academy"
            ],
            "Planning and Budgeting DHIS2 Implementations": ["Planning and Budgeting DHIS2 Implementations"]
        }

        # Create reverse mapping for quick lookups
        self.reverse_mapping = {}
        for standard_name, variations in self.course_mapping.items():
            for variant in variations:
                self.reverse_mapping[variant] = standard_name

    def get_standard_name(self, course_name):
        """Convert any course name variation to its standard form"""
        if pd.isna(course_name):
            return course_name
        
        course_name = str(course_name).strip()
        return self.reverse_mapping.get(course_name, course_name)

    def standardize_dataframe(self, df, column_name='Course Name'):
        """Standardize course names in a dataframe"""
        if column_name in df.columns:
            df[column_name] = df[column_name].apply(self.get_standard_name)
        return df

    def verify_course_names(self, df, column_name='Course Name'):
        """Verify and report course name issues in a dataframe"""
        unknown_variants = set()
        counts = {'standardized': 0, 'unknown': 0, 'total': 0}
        
        if column_name in df.columns:
            for course_name in df[column_name].dropna().unique():
                counts['total'] += 1
                if course_name in self.reverse_mapping:
                    counts['standardized'] += 1
                else:
                    counts['unknown'] += 1
                    unknown_variants.add(course_name)
        
        return unknown_variants, counts