import pandas as pd
import sciris as sc

screening = sc.loadobj('nigeria_screening_results_novaccine.obj')
screening.to_excel('nigeria_screening_results_novx_LT.xlsx')



