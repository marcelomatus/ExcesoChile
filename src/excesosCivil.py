#encoding=utf-8 
'''
MIT License

Copyright (c) 2020 Sebastian Cornejo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os

import numpy as np
import pandas as pd
import glob
import numpy as np
import requests
import datetime as dt
import io
import sys

#
# Utils
#
def regionNameRegex(df):
    df['Region'] = df['Region'].replace(regex=True, to_replace=r'.*Región de ', value=r'')
    df['Region'] = df['Region'].replace(regex=True, to_replace=r'.*Región del ', value=r'')

def normalizaNombreCodigoRegionYComuna(df):
    # standards:
    df["Comuna"] = df["Comuna"].replace({"Coyhaique": "coihaique",
                                         "Paihuano": "paiguano",
                                         "La Calera": "Calera",
                                         "Llay-Llay": "Llaillay",
                                         })

    # Lee IDs de comunas desde página web oficial de SUBDERE
    df_dim_comunas = pd.read_excel("http://www.subdere.gov.cl/sites/default/files/documentos/cut_2018_v03.xls",
                                   encoding="utf-8")

    # Crea columna sin tildes, para hacer merge con datos publicados
    #df_dim_comunas["Comuna"] = df_dim_comunas["Nombre Comuna"].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    df_dim_comunas["Comuna"] = df_dim_comunas["Nombre Comuna"].str.normalize("NFKD")\
        .str.encode("ascii", errors="ignore").str.decode("utf-8").str.lower().str.replace(' ', '')


    df["Comuna"] = df["Comuna"].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")\
        .str.lower().str.replace(' ', '')

    #df = df.merge(df_dim_comunas, on="Comuna", how="outer")
    df = df.merge(df_dim_comunas, on="Comuna", how="inner")

    df['Comuna'] = df['Nombre Comuna']
    df['Codigo comuna'] = df['Código Comuna 2017']
    df['Region'] = df['Nombre Región']
    df['Codigo region'] = df['Código Región']

    df.drop(columns={'Código Región','Nombre Región',
                     'Código Comuna 2017', 'Nombre Comuna',
                     'Código Provincia', 'Nombre Provincia'
                     }, inplace=True)

    # Sort Columns
    columnsAddedHere = ['Region', 'Codigo region', 'Comuna', 'Codigo comuna']
    originalColumns = [x for x in list(df) if x not in columnsAddedHere]
    sortedColumns = columnsAddedHere + originalColumns

    #report on missing
    df1 = df[df.isnull().any(axis=1)]
    if df1.size > 0:
        print(df1.to_string())

    df = df[sortedColumns]
    df['Codigo region'] = df['Codigo region'].astype(str)
    return df


#
# Exceso
#

def regionName16(df):
    df["Region"] = df["Region"].replace({"Arica - Parinacota": "Arica y Parinacota",
                                         "Tarapaca": "Tarapacá", "Valparaiso": "Valparaíso",
                                         "Región Metropolitana de Santiago": "Metropolitana",
                                         "Del Libertador General Bernardo O’Higgins": "O’Higgins",
                                         "Libertador General Bernardo OHiggins": "O’Higgins",
                                         "Nuble": "Biobío y Ñuble",
                                         "Ñuble": "Biobío y Ñuble",
                                         "Biobio": "Biobío y Ñuble",
                                         "Biobío": "Biobío y Ñuble",
                                         "Concepción": "Biobío y Ñuble",
                                         "La Araucania": "Araucanía", "la Araucanía": "Araucanía",
                                         "Los Rios": "Los Ríos", "de Los Ríos": "Los Ríos", "De los Rios": "Los Ríos",
                                         "De los Lagos": "Los Lagos",
                                         "Aysen": "Aysén", "Aysén del General Carlos Ibañez del Campo": "Aysén",
                                         "Magallanes y la Antartica": "Magallanes",
                                         "Magallanes y de la Antártica Chilena": "Magallanes"
                                         })

    df['Codigo region'] = df['Codigo region'].replace('16', '8')
    

def normalizeRegCivDF(df):
    # rename columns
    df.rename(columns={'REGION': 'Region', 'COMUNA': 'Comuna'}, inplace=True)
    # title case => Title Case
    df['Comuna'] = df['Comuna'].str.title()
    regionNameRegex(df)

    # zero pad fechas
    df['MES'] = df['MES'].astype(str).apply(lambda x: x.zfill(2))
    df['DIA'] = df['DIA'].astype(str).apply(lambda x: x.zfill(2))
    # standard fecha
    df["Fecha"] = df["AÑO"].astype(str) + '-' + df["MES"].astype(str) + '-' + df["DIA"].astype(str)
    df = df.drop(columns={'AÑO', 'MES', 'DIA'})

    df = normalizaNombreCodigoRegionYComuna(df)
    df.sort_values(by=['Codigo region', 'Codigo comuna', 'Fecha'], na_position='first', inplace=True)
    return df


def dayofyear(x):
    d = dt.date.fromisoformat(x)
     
    return ((d - dt.date(d.year,1,1)).days, d.year)

def weekofyear(x, diff = 0):
    (doy, year) = dayofyear(x)
    return  (int((doy - diff) / 7) + 1, year)

import statistics as st


import numpy as np
import scipy
import scipy.stats


def mean_std_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, sd = np.mean(a), np.std(a, ddof=1)
    if sd == 0:
        return 0, 0, 0, 0

    ht = scipy.stats.t.interval(confidence, df=n-1, loc=m, scale= sd / np.sqrt(n) )
    hn = scipy.stats.norm.interval(confidence, loc=m, scale= sd / np.sqrt(n) )
    return m, sd, 0.5*(hn[1] - hn[0]), 0.5*(ht[1] - ht[0])

def updateExceso(fte, 
                 fromDate = '2015-01-01',
                 toDate   = dt.datetime.today().strftime("%Y-%m-%d"),
                 confidence = 0.95,
                 contagio = dt.date(2020,3,2)):

    outputPrefix = 'excesos'

    # get the xlsx from the API
    headers = {
        'Content-Type': 'application/json',
        'Origin': fte.replace('/api/estadistica/', ''),
        'Connection': 'keep-alive',
    }
    myData = {
        "startdate": fromDate,
        "enddate": toDate
    }

    suffix = 'defuncion'
    call = fte + suffix + '/getXlsxAllComunas'

    fileName = fromDate + '_' + toDate 

    # load the API to a DF
    if False:
        df_API = pd.read_csv(suffix + '_' + fileName + '_ori.csv')
    else:
        print('Querying ' + call + ' between ' + fromDate + ' and ' + toDate)
        response = requests.post(call, headers=headers, json=myData)
        xlsx = io.BytesIO(response.content)
        df_API = pd.read_excel(xlsx)
        df_API.to_csv(suffix + '_' + fileName + '_ori.csv', index=False)
        
    df_API = normalizeRegCivDF(df_API)
        
    # Normalize and sort
    df_API = normalizaNombreCodigoRegionYComuna(df_API)
    df_API.sort_values(by=['Codigo region', 'Codigo comuna', 'Fecha'], inplace=True)
    df_API.rename(columns={'TOTAL': 'Defunciones'}, inplace=True)
    
    regionName16(df_API)
    
    df_API.to_csv(suffix + '_' + fileName + '_nor.csv', index=False)
    

    print('starting calc')
    ddays = (contagio - dt.date(2020,1,1)).days
    diff = ddays - int(ddays/7)*7

    num_weeks = 53
    num_years = 6
    num_reg = 17
    defunciones = np.zeros((num_reg+1,num_weeks+1,num_years+1))
    PUCA = np.zeros((num_reg, num_weeks))
    DUCA = np.zeros((num_reg, num_weeks))
    HCIN = np.zeros((num_reg, num_weeks))
    HCIT = np.zeros((num_reg, num_weeks))

    nombreRegion = {}
    max_reg = 0
    max_week = 0
    for index, row in df_API.iterrows():
        (week, year) = weekofyear(row['Fecha'], diff)
        reg = int(row['Codigo region'])
        reg = 8 if reg == 16 else reg
        nombreRegion[reg] = row['Region']
        defun = int(row['Defunciones'])

        defunciones[reg][week][year - 2015] += defun
        defunciones[0][week][year - 2015] += defun
        max_reg = max(max_reg, reg)
        max_week = max(max_week, week)


    for r in range(0,max_reg + 1):
        for w in range(0, max_week + 1):
            (puca, duca, hn, ht) = mean_std_confidence_interval(defunciones[r][w][0:5])
            PUCA[r][w] = puca
            DUCA[r][w] = duca
            HCIN[r][w] = hn
            HCIT[r][w] = ht
            

    columns=('Nombre Region',
             'Region',
             'Semana',
             'Comienzo',
             'Fin',
             '2015'  ,
             '2016'  ,
             '2017'  ,
             '2018'  ,
             '2019'  ,
             '2020'  ,
             'PUCA'  ,
             'DUCA'  ,
             'Exceso',
             'p_factor',
             'z_factor',
             'ExcesoN',
             'p_factorN',
             'z_factorN',
             'ExcesoT',
             'p_factorT',
             'z_factorT',
         )
    
    edf = pd.DataFrame(columns=columns)
    nombreRegion[0] = 'Chile'
    for r in range(0,max_reg+1):
        for w in range(1, max_week+1):
            puca = PUCA[r][w]
            duca = DUCA[r][w]
            hn = HCIN[r][w]
            ht = HCIT[r][w]
            def2020 = defunciones[r][w][5]
            if def2020:
                exceso    = def2020 - puca
                p_factor  = exceso/puca if puca != 0 else 0
                z_factor  = exceso/duca if duca != 0 else 0
                excesoN   = exceso - hn if exceso > hn else exceso + hn if exceso < -hn else 0.0
                p_factorN = excesoN/puca if puca != 0 else 0
                z_factorN = excesoN/duca if duca != 0 else 0
                excesoT   = exceso - ht if exceso > ht else exceso + ht if exceso < -ht else 0.0
                p_factorT = excesoT/puca if puca != 0 else 0
                z_factorT = excesoT/duca if duca != 0 else 0
            else:
                exceso    = ''
                p_factor  = ''
                z_factor  = ''
                excesoN   = ''
                p_factorN = ''
                z_factorN = ''
                excesoT   = ''
                p_factorT = ''
                z_factorT = ''
                
            
            comienzo = dt.date(2020,1, 1) + dt.timedelta(diff + (w - 1)*7 )
            fin = comienzo + dt.timedelta(6)
            row = {'Nombre Region'  : nombreRegion[r],
                   'Region'  : r,
                   'Semana'  : w,
                   'Comienzo' : comienzo.strftime("%Y-%m-%d"),
                   'Fin' : fin.strftime("%Y-%m-%d"),
                   '2015'  : defunciones[r][w][0],
                   '2016'  : defunciones[r][w][1],
                   '2017'  : defunciones[r][w][2],
                   '2018'  : defunciones[r][w][3],
                   '2019'  : defunciones[r][w][4],
                   '2020'  : defunciones[r][w][5],
                   'PUCA'  : puca,
                   'DUCA'  : duca,
                   'Exceso'   : exceso,
                   'p_factor' : p_factor,
                   'z_factor' : z_factor,
                   'ExcesoN'  : excesoN,
                   'p_factorN': p_factorN,
                   'z_factorN': z_factorN,
                   'ExcesoT'  : excesoT,
                   'p_factorT': p_factorT,
                   'z_factorT': z_factorT,
            }
            edf = edf.append(row, ignore_index=True)
    
    
    print('end calc')
    # check for duplicates:


    edf = edf.astype({'Nombre Region': str,
                      'Region': int,
                      'Semana': int,
                      'Comienzo' : str,
                      'Fin' : str,
                      '2015'  : float,
                      '2016'  : float,
                      '2017'  : float,
                      '2018'  : float,
                      '2019'  : float,
                      '2020'  : float,
                      'PUCA'  : float,
                      'DUCA'  : float,
                      'Exceso':  str,
                      'p_factor': str,
                      'z_factor': str,
                      'ExcesoN': str,
                      'p_factorN': str,
                      'z_factorN': str,
                      'ExcesoT': str,
                      'p_factorT': str,
                      'z_factorT': str,
                  })
    edf.to_csv(outputPrefix + '_' + fileName + '.csv', index=False, columns=columns)



if __name__ == '__main__':

    URL = 'https://api.sed.srcei.cl/api/estadistica/'
    contagio = dt.date(2020,3,2)    
    confidence = 0.95
    
    if len(sys.argv) == 3:
        fromDate=sys.argv[1]
        toDate=sys.argv[2]
    elif len(sys.argv) == 2:
        td = dt.datetime.today()
        fromDate='2015-01-01'
        toDate=sys.argv[1]
    elif len(sys.argv) == 1:
        td = dt.datetime.today()
        fromDate='2015-01-01'
        toDate=td.strftime("%Y-%m-%d")
    else:
        print('something\'s wrong with ' + str(len(sys.argv)) + ' arguments')

    print('Actualizando excesos entre ' + fromDate + ' y ' + toDate)
    updateExceso(URL,
                 fromDate=fromDate,
                 toDate=toDate,
                 contagio=contagio,
                 confidence=0.95)
        
    # removeOldFiles()
    # test_compareAPIAgainstFile()
