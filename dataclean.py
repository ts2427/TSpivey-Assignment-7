import pandas as pd

def add_databreach_data(filename):
    """
    Reads in the specified Data Breach Excel file, cleans it by removing unnecessary rows,
    and returns a cleaned dataframe.
    """
    df_databreach = pd.read_excel(filename)
    cols_we_need = ['id', 'org_name',
       'reported_date', 'breach_date', 'end_breach_date', 'incident_details',
       'information_affected',
       'organization_type',
       'breach_type',
       'normalized_org_name',
       'group_org_breach_type',
       'group_org_type',
       'total_affected', 'residents_affected',
       'breach_location_street',
       'breach_location_city', 'breach_location_state', 'breach_location_zip',
       'breach_location_country', 'tags',
       ]
    df_databreach_clean = df_databreach[df_databreach.breach_type != 'UNKN']
    df_databreach_clean = df_databreach_clean[cols_we_need]
    return df_databreach_clean

def universal_clean():

    """
    Cleans multiple datasets and returns a dictionary of cleaned dataframes.
    """
    df_databreach = add_databreach_data('Data_Breach_Chronology.xlsx')

    full_dictionary_file = {
        'databreach': df_databreach,
    }
    return full_dictionary_file

