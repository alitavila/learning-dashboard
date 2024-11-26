import pandas as pd
import os
from datetime import datetime
from utils import standardize_course_name  # Import our course name standardization

def merge_jira_data(existing_file='jira_support_tickets.csv', new_file=None, backup=True):
    """
    Merge new JIRA export data with existing support tickets data.
    Handles multiple files and prevents duplicates.
    
    Parameters:
    existing_file (str): Path to existing JIRA data CSV file
    new_file (str): Path to new JIRA export CSV file (or directory)
    backup (bool): Whether to create a backup of existing file
    
    Returns:
    tuple: (success (bool), message (str))
    """
    try:
        # Create backup
        if backup and os.path.exists(existing_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f'backup_{timestamp}_{existing_file}'
            os.rename(existing_file, backup_file)
            print(f"Backup created: {backup_file}")
        
        # Read existing data
        try:
            existing_df = pd.read_csv(backup_file if backup else existing_file, 
                                    encoding='ISO-8859-1')
            print(f"Existing data loaded: {len(existing_df)} rows")
        except FileNotFoundError:
            existing_df = pd.DataFrame()
            print("No existing data found, will create new file")
        
        # Standardize existing course names if data exists
        if not existing_df.empty and 'Course Name' in existing_df.columns:
            existing_df['Course Name'] = existing_df['Course Name'].apply(standardize_course_name)
        
        # Handle new data files
        new_dfs = []
        
        if new_file and os.path.isfile(new_file):
            # Single file specified
            files_to_process = [new_file]
        else:
            # Look for all jira export files
            files_to_process = [f for f in os.listdir() if f.startswith('jira_export_')]
        
        if not files_to_process:
            return False, "No new export files found"
        
        # Process each new file
        for file in files_to_process:
            try:
                df = pd.read_csv(file, encoding='ISO-8859-1')
                # Standardize course names in new data
                if 'Course Name' in df.columns:
                    df['Course Name'] = df['Course Name'].apply(standardize_course_name)
                new_dfs.append(df)
                print(f"Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if not new_dfs:
            return False, "Failed to load any new data"
        
        # Combine all new data
        new_df = pd.concat(new_dfs, ignore_index=True)
        
        # Convert date columns to datetime
        date_columns = ['Created', 'Updated']
        for df in [existing_df, new_df]:
            if not df.empty:
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Combine existing and new data
        if existing_df.empty:
            combined_df = new_df
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates based on multiple criteria
        combined_df = combined_df.drop_duplicates(
            subset=[
                'Issue Type',
                'Category',
                'Course Name',
                'Created',
                'Assignee Id'  # Added to help identify unique tickets
            ],
            keep='last'
        )
        
        # Sort by Created date
        combined_df = combined_df.sort_values('Created')
        
        # Save merged data
        combined_df.to_csv(existing_file, index=False, encoding='ISO-8859-1')
        
        # Print detailed summary
        print("\nMerge Summary:")
        if not existing_df.empty:
            new_records = len(combined_df) - len(existing_df)
            print(f"Previous records: {len(existing_df)}")
            print(f"New records added: {new_records}")
        print(f"Total records: {len(combined_df)}")
        
        # Date range info
        print("\nDate range:")
        print(f"First record: {combined_df['Created'].min()}")
        print(f"Last record: {combined_df['Created'].max()}")
        
        # Course distribution
        print("\nTickets per course:")
        course_counts = combined_df['Course Name'].value_counts()
        for course, count in course_counts.items():
            print(f"{course}: {count}")
        
        return True, f"Successfully merged data. Total records: {len(combined_df)}"
        
    except Exception as e:
        # If error occurs, restore backup if it exists
        if backup and 'backup_file' in locals() and os.path.exists(backup_file):
            os.rename(backup_file, existing_file)
            return False, f"Error occurred, backup restored: {str(e)}"
        return False, f"Error occurred: {str(e)}"

if __name__ == "__main__":
    import sys
    
    # Allow specifying a specific file as argument
    new_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Starting JIRA data merge...")
    success, message = merge_jira_data(new_file=new_file)
    print(message)