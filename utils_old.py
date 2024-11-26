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
                "TEST Introduction Ã  DHIS2"
            ],
            "Les principes fondamentaux de DHIS2": [
                "Les principes fondamentaux de DHIS2",
                "Les principes fondamentaux de  DHIS2"
            ],
            "Les fondamentaux de DHIS2 événements": [
                "Les fondamentaux de DHIS2 evenements",
                "Les fondamentaux de DHIS2 événements",
                "Les fondamentaux de DHIS2 ÃvÃ©nements",
                "Les fondamentaux de DHIS2 Événements",
                "Les fondamentaux de DHIS2 ÃƒÂ©vÃƒÂ©nements"
            ],
            "Le paramétrage de DHIS2 agrégé": [
                "Le parametrage de DHIS2 agrege",
                "Le paramétrage de DHIS2 agrégé",
                "Le paramÃ©trage de DHIS2 agrÃ©gÃ©"
            ],
            "Les fondamentaux de la saisie et de la validation des données agrégées": [
                "Les fondamentaux de la saisie et de la validation des données agrégées"
            ],
            "Académie DHIS2 Numérisée de Niveau 1 - Outils d'Analyse": [
                "AcadÃ©mie DHIS2 NumÃ©risÃ©e de Niveau 1 - Outils d'Analyse"
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
            "Planning and Budgeting DHIS2 Implementations": ["Planning and Budgeting DHIS2 Implementations"],

            # Additional Courses (might need to be excluded from main dashboard)
            "DHIS2 Analytics Tools Academy": [
                "DHIS2 Analytics Tools Academy Level 1",
                "DHIS2 Analytics Tools Academy Level 1 "
            ],
            "DHIS2 Android Academy": [
                "DHIS2 Android Implementation Academy Level 2",
                "DHIS2 Android Implementers Academy"
            ],
            "DHIS2 Tracker Academy": [
                "DHIS2 Tracker Level 1"
            ],
            "Design for Analytics Academy": [
                "Design for Analytics Level 1 - East Africa"
            ],
            "DHIS2 Indonesia": [
                "DHIS2 Fundamental Indonesia  - Angkatan 2",
                "DHIS2 Fundamental Indonesia - Angkatan 1",
                "DHIS2 Introduction for Beginers (Indonesia)"
            ],
            
            # Test/Support courses (should probably be excluded)
            "TEST": [
                "Support Test",
                "Testing course"
            ]
        }

        # List of courses to exclude from dashboard
        self.excluded_courses = {
            "DHIS2 Analytics Tools Academy",
            "DHIS2 Android Academy",
            "DHIS2 Tracker Academy",
            "Design for Analytics Academy",
            "DHIS2 Indonesia",
            "TEST",
            "Other"  # Special case in support tickets
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

    def is_excluded_course(self, course_name):
        """Check if a course should be excluded from the dashboard"""
        standard_name = self.get_standard_name(course_name)
        return standard_name in self.excluded_courses or course_name == "Other"

    def standardize_dataframe(self, df, column_name='Course Name'):
        """Standardize course names in a dataframe and remove excluded courses"""
        if column_name in df.columns:
            # First standardize all names
            df[column_name] = df[column_name].apply(self.get_standard_name)
            # Then filter out excluded courses
            if column_name != 'Other':  # Keep 'Other' for support tickets
                df = df[~df[column_name].apply(self.is_excluded_course)]
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