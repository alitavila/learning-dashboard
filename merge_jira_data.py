import pandas as pd
import os
from datetime import datetime

def merge_jira_data(existing_file='jira_support_tickets.csv', new_file=None, backup=True):
    """
    Merge new JIRA export data with existing support tickets data.
    
    Parameters:
    existing_file (str): Path to existing JIRA data CSV file
    new_file (str): Path to new JIRA export CSV file
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
        
        # Read new data
        if not new_file:
            # Find the most recent jira_export file in the current directory
            export_files = [f for f in os.listdir() if f.startswith('jira_export_')]
            if not export_files:
                return False, "No new export file found"
            new_file = max(export_files)  # Gets the most recent file by name
        
        new_df = pd.read_csv(new_file, encoding='ISO-8859-1')
        print(f"New data loaded: {len(new_df)} rows")
        
        # Convert date columns to datetime
        date_columns = ['Created', 'Updated']
        
        # Define el orden deseado de las columnas
        column_order = ['Issue Type', 'Category', 'Course Name', 'Assignee', 
                       'Assignee Id', 'Status', 'Created', 'Updated']
        
        # Estandarizar orden de columnas en ambos dataframes
        if not existing_df.empty:
            existing_df = existing_df[column_order]
            for col in date_columns:
                existing_df[col] = pd.to_datetime(existing_df[col], format='%m/%d/%Y %H:%M')
        
        if not new_df.empty:
            new_df = new_df[column_order]
            for col in date_columns:
                new_df[col] = pd.to_datetime(new_df[col], format='%d/%m/%Y %H:%M')

        # Combine data
        if existing_df.empty:
            combined_df = new_df
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Remove duplicates based on key fields
        combined_df = combined_df.drop_duplicates(
            subset=['Course Name', 'Category', 'Created', 'Assignee'],
            keep='last'
        )

        # Sort by Created date
        combined_df = combined_df.sort_values('Created')

        # Ensure final dataframe has correct column order
        combined_df = combined_df[column_order]
        
        # Save merged data
        combined_df.to_csv(existing_file, index=False, encoding='ISO-8859-1')
        
        # Print summary
        if not existing_df.empty:
            new_records = len(combined_df) - len(existing_df)
            print(f"\nMerge Summary:")
            print(f"Previous records: {len(existing_df)}")
            print(f"New records added: {new_records}")
            print(f"Total records: {len(combined_df)}")
            
            # Date range info
            print(f"\nDate range:")
            print(f"First record: {combined_df['Created'].min()}")
            print(f"Last record: {combined_df['Created'].max()}")
        
        return True, f"Successfully merged data. Total records: {len(combined_df)}"
        
    except Exception as e:
        # If error occurs, restore backup if it exists
        if backup and os.path.exists(backup_file):
            os.rename(backup_file, existing_file)
            return False, f"Error occurred, backup restored: {str(e)}"
        return False, f"Error occurred: {str(e)}"

if __name__ == "__main__":
    # Example usage
    success, message = merge_jira_data()
    print(message)
    