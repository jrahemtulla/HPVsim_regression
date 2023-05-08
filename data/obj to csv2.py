import pandas as pd
import sciris as sc
import openpyxl


vaccination_coverage = sc.loadobj('vaccination_coverage.obj')
df = pd.DataFrame.from_dict(vaccination_coverage)

# Save the DataFrame as an Excel file
df.to_excel('vaccination_coverage.xls', index=False)


