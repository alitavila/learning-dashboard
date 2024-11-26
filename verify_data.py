import pandas as pd
from course_mapper import CourseNameMapper

def verify_all_data():
    # Initialize course mapper
    mapper = CourseNameMapper()
    
    # Load all data sources
    data_files = {
        'Enrollments': 'all_enrollments.csv',
        'Certificates': 'all_certificates.csv',
        'Support Tickets': 'jira_support_tickets.csv',
        'Course List': 'About the course.xlsx'
    }
    
    print("Loading and verifying all data sources...")
    
    for source_name, filename in data_files.items():
        print(f"\n=== {source_name} ===")
        try:
            # Load data
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename, encoding='ISO-8859-1')
            
            # Verify course names before standardization
            print("\nBefore standardization:")
            unknown_variants = mapper.verify_course_names(df)
            
            # Apply standardization
            df_standardized = mapper.standardize_dataframe(df)
            
            # Get course statistics
            if 'Course Name' in df.columns:
                print("\nCourse Statistics:")
                course_counts = df_standardized['Course Name'].value_counts()
                print("\nEnrollments per course:")
                for course, count in course_counts.items():
                    print(f"{course}: {count:,}")
                
                # For enrollments, show recent counts
                if source_name == 'Enrollments':
                    print("\nLast 30 days enrollments per course:")
                    df_standardized['Date Joined'] = pd.to_datetime(df_standardized['Date Joined'])
                    recent_mask = df_standardized['Date Joined'] >= pd.Timestamp.now() - pd.Timedelta(days=30)
                    recent_counts = df_standardized[recent_mask]['Course Name'].value_counts()
                    for course, count in recent_counts.items():
                        print(f"{course}: {count:,}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    verify_all_data()