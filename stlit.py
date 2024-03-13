import streamlit as st
import pandas as pd
import streamlit_pandas as sp
from streamlit_shap import st_shap
import joblib
import shap

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns;sns.set()
import numpy as np


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def load_data():
    df = pd.read_excel('2023_Data_BiomassGasification_NED_Stlit.xlsx','treated')

    return df

st.title("Machine Learning for Predictive Modelling and Optimization of Hydrogen Production from Biomass Gasification")


st.divider()
#---------------------------------------datail-------------------------------------------------#
st.subheader("Introduction")
st.write("Due to the complexity involved in calculating the hydrogen gas generated from biomass gasification "
         "processes using traditional chemical methods, a new approach has been devised. This involves utilizing "
         "machine learning to create a predictive model for estimating the hydrogen gas produced. "
         "To enhance user-friendliness, a web platform has been developed to allow users to experiment with and "
         "utilize this predictive model effectively.")
st.divider()
st.markdown(
    """
    <style>
        .custom-divider {
            border-top: 0.5px dashed #555;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True, )
#input
st.header("	:round_pushpin: Input For Hydrogen Prediction")
st.subheader("Feed Stock Information")

col1,col2,col3,col4=st.columns(4)
with col1:
    Type = st.selectbox('Type of feedstock', ('woody biomass', 'herbaceous biomass', 'plastics', 'sewage sludge', 'municipal solid waste', 'other'))
with col2:
    Shape = st.selectbox('Shape of feedstock', ('particle', 'chips', 'dust', 'fibres', 'pellets', 'other'))
with col3:
    Particle_size = st.number_input("Particle size (mm)", placeholder="Type a number...")
with col4:
    LHV = st.number_input("Low heating values (MJ/kg)", placeholder="Type a number...")
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.subheader("Ultimate Analysis")
col5,col6,col7,col8,col9=st.columns(5)
with col5:
    Carbon = st.number_input("Percent of Carbon", placeholder="Type a number...")
with col6:
    Hydrogen = st.number_input("Percent of Hydrogen", placeholder="Type a number...")
with col7:
    Oxygen = st.number_input("Percent of Oxygen", placeholder="Type a number...")
with col8:
    Nitrogen = st.number_input("Percent of Nitrogen", placeholder="Type a number...")
with col9:
    Sulfur = st.number_input("Percent of Sulfur", placeholder="Type a number...")
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.subheader("Proximate Analysis")
col10,col11,col12,col13=st.columns(4)
with col10:
    Moisture = st.number_input("Moisture (%)", placeholder="Type a number...")
with col11:
    Ash = st.number_input("Ash (%)", placeholder="Type a number...")
with col12:
    Volatile = st.number_input("Volatile matter (%)", placeholder="Type a number...")
with col13:
    Fixed_carbon = st.number_input("Fixed carbon (%)", placeholder="Type a number...")
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.subheader("Ligocellulosic Composition")
col14,col15,col16=st.columns(3)
with col14:
    Cellulose = st.number_input("Cellulose (% dry basis)", placeholder="Type a number...")
with col15:
    Hemicellulose = st.number_input("Hemicellulose (% dry basis)", placeholder="Type a number...")
with col16:
    Lignin = st.number_input("Lignin (% dry basis)", placeholder="Type a number...")
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.subheader("Operation")
col17,col18,col19,col20=st.columns(4)
with col17:
    gasifying = st.selectbox('Gasifying agent', ('air', 'air+steam', 'oxygen', 'steam+oxygen', 'steam', 'other'))
    Res_time = st.number_input("Residence time (min)", placeholder="Type a number...")
with col18:
    Operating_con = st.selectbox('Operating condition', ('continuous', 'batch', 'other'))
    S_B=st.number_input("Steam/Biomass ratio", placeholder="Type a number...")
with col19:
    Operating_p = st.selectbox('Operating pressure (kPa)', ('atmospheric', '111.46', '205', 'slightly above atmospheric', 'slightly below atmospheric', 'other'))
    ER = st.number_input("Equivalent ratio", placeholder="Type a number...")
with col20:
    Temp=st.number_input("Temperature (C)", placeholder="Type a number...")
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.subheader("Reactor")
col24,col25,col26,col27=st.columns(4)
with col24:
    reactor = st.selectbox('Reactor Types', ('fixed bed', 'fluidised bed', 'horizontal tube', 'other bed'))
with col25:
    bed = st.selectbox('Bed material',('silica', 'sand', 'alumina', 'dolomite', 'olivine','Y-alumina','calcium oxide'))
with col26:
    Scale = st.selectbox('Scale', ('pilot', 'lab', "other"))
with col27:
    cat= st.selectbox('Catalyst',('Yes', 'No ',))
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
#NAN onditon particle size
if Particle_size == 0 and Type == "woody biomass" and Shape == "pellets":
    Particle_size = 9.35
if Particle_size == 0 and Type == "woody biomass" and Shape == "chips":
    Particle_size = 7.90
if Particle_size == 0 and Type == "woody biomass" and Shape == "dust":
    Particle_size = 0.5
if Particle_size == 0 and Type == "woody biomass" and Shape == "fibres":
    Particle_size = 4
if Particle_size == 0 and Type == "woody biomass" and Shape == "particle":
    Particle_size = 1.25
if Particle_size == 0 and Type == "woody biomass" and Shape == "other":
    Particle_size = 1.38
if Particle_size == 0 and Type == "herbaceous biomass" and Shape == "pellets":
    Particle_size = 2.5
if Particle_size == 0 and Type == "herbaceous biomass" and Shape == "chips":
    Particle_size = 9.35
if Particle_size == 0 and Type == "herbaceous biomass" and Shape == "dust":
    Particle_size = 0.5
if Particle_size == 0 and Type == "herbaceous biomass" and Shape == "fibres":
    Particle_size = 4
if Particle_size == 0 and Type == "herbaceous biomass" and Shape == "particle":
    Particle_size = 3.75
if Particle_size == 0 and Type == "herbaceous biomass" and Shape == "other":
    Particle_size = 22.62
if Particle_size== 0 and Type=="plastics" and  Shape== "pellets":
    Particle_size=2
if Particle_size== 0 and Type=="plastics" and  Shape== "chips":
    Particle_size=7.9
if Particle_size== 0 and Type=="plastics" and  Shape== "dust":
    Particle_size=0.5
if Particle_size== 0 and Type=="plastics" and  Shape== "fibres":
    Particle_size=4
if Particle_size== 0 and Type=="plastics" and  Shape== "particle":
    Particle_size=1.75
if Particle_size== 0 and Type=="plastics" and  Shape== "other":
    Particle_size=3.5
if Particle_size == 0 and Type == "municipal solid waste"and Shape== "pellets":
    Particle_size = 5
if Particle_size == 0 and Type == "municipal solid waste"and Shape== "chips":
    Particle_size = 5
if Particle_size == 0 and Type == "municipal solid waste"and Shape== "dust":
    Particle_size = 5
if Particle_size == 0 and Type == "municipal solid waste"and Shape== "fibres":
    Particle_size = 5
if Particle_size == 0 and Type == "municipal solid waste"and Shape== "particle":
    Particle_size = 1
if Particle_size == 0 and Type == "municipal solid waste" and Shape == "other":
    Particle_size = 5
if Particle_size == 0 and Type == "sewage sludge"and Shape== "pellets":
    Particle_size = 6
if Particle_size == 0 and Type == "sewage sludge"and Shape== "chips":
    Particle_size = 7.9
if Particle_size == 0 and Type == "sewage sludge"and Shape== "dust":
    Particle_size = 0.5
if Particle_size == 0 and Type == "sewage sludge"and Shape== "fibres":
    Particle_size = 4
if Particle_size == 0 and Type == "sewage sludge"and Shape== "particle":
    Particle_size = 3.5
if Particle_size == 0 and Type == "sewage sludge" and Shape == "other":
    Particle_size = 3.5
if Particle_size == 0 and Type == "other"and Shape== "pellets":
    Particle_size = 6
if Particle_size == 0 and Type == "other"and Shape== "chips":
    Particle_size = 5
if Particle_size == 0 and Type == "other"and Shape== "dust":
    Particle_size = 0.5
if Particle_size == 0 and Type == "other"and Shape== "fibres":
    Particle_size = 4
if Particle_size == 0 and Type == "other"and Shape== "particle":
    Particle_size = 1.75
if Particle_size == 0 and Type == "other" and Shape == "other":
    Particle_size = 3.5
if Particle_size == 0 and Type == "woody biomass":
    Particle_size = 6
if Particle_size == 0 and Type == "sewage sludge":
    Particle_size = 3.5
if Particle_size == 0 and Type == "plastics":
    Particle_size = 2
if Particle_size == 0 and Type == "other":
    Particle_size = 5
if Particle_size == 0 and Type == 'municipal solid waste':
    Particle_size = 5
if Particle_size == 0 and Type == 'herbaceous biomass':
    Particle_size = 9.35

#S_B
if S_B == 0 and Type == "woody biomass" and Shape == "pellets":
    S_B = 0.18
if S_B == 0 and Type == "woody biomass" and Shape == "chips":
    S_B = 0.8
if S_B == 0 and Type == "woody biomass" and Shape == "dust":
    S_B = 0.3
if S_B == 0 and Type == "woody biomass" and Shape == "fibres":
    S_B = 1.05
if S_B == 0 and Type == "woody biomass" and Shape == "particle":
    S_B = 0.3
if S_B == 0 and Type == "woody biomass" and Shape == "other":
    S_B = 1.2
if S_B == 0 and Type == "woody biomass":
    s_B=0.55
if S_B == 0 and Type == "herbaceous biomass" and Shape == "pellets":
    S_B = 0.22
if S_B == 0 and Type == "herbaceous biomass" and Shape == "chips":
    S_B = 0.5
if S_B == 0 and Type == "herbaceous biomass" and Shape == "dust":
    S_B = 0.5
if S_B == 0 and Type == "herbaceous biomass" and Shape == "fibres":
    S_B = 0.5
if S_B == 0 and Type == "herbaceous biomass" and Shape == "particle":
    S_B = 0.8
if S_B == 0 and Type == "herbaceous biomass" and Shape == "other":
    S_B = 0.5
if S_B == 0 and Type == "herbaceous biomass":
    s_B=0.48
if S_B == 0 and Type == "plastics" and Shape == "pellets":
    S_B = 0.48
if S_B == 0 and Type == "plastics" and Shape == "chips":
    S_B = 0.8
if S_B == 0 and Type == "plastics" and Shape == "dust":
    S_B = 1.56
if S_B == 0 and Type == "plastics" and Shape == "fibres":
    S_B = 0.5
if S_B == 0 and Type == "plastics" and Shape == "particle":
    S_B = 1.16
if S_B == 0 and Type == "plastics" and Shape == "other":
    S_B = 1.2
if S_B == 0 and Type == "plastics":
    s_B=0.98
if S_B == 0 and Type == "municipal solid waste" and Shape == "pellets":
    S_B = 1.16
if S_B == 0 and Type == "municipal solid waste" and Shape == "chips":
    S_B = 1.16
if S_B == 0 and Type == "municipal solid waste" and Shape == "dust":
    S_B = 1.16
if S_B == 0 and Type == "municipal solid waste" and Shape == "fibres":
    S_B = 1.16
if S_B == 0 and Type == "municipal solid waste" and Shape == "particle":
    S_B = 1.16
if S_B == 0 and Type == "municipal solid waste" and Shape == "other":
    S_B = 1.16
if S_B == 0 and Type == "municipal solid waste":
    s_B= 1.16
if S_B == 0 and Type == "sewage sludge" and Shape == "pellets":
    S_B = 0.22
if S_B == 0 and Type == "sewage sludge" and Shape == "chips":
    S_B = 0.8
if S_B == 0 and Type == "sewage sludge" and Shape == "dust":
    S_B = 1.56
if S_B == 0 and Type == "sewage sludge" and Shape == "fibres":
    S_B = 0.5
if S_B == 0 and Type == "sewage sludge" and Shape == "particle":
    S_B = 0.67
if S_B == 0 and Type == "sewage sludge" and Shape == "other":
    S_B = 0.75
if S_B == 0 and Type == "sewage sludge":
    s_B= 0.71
if S_B == 0 and Type == "other" and Shape == "pellets":
    S_B = 0.6
if S_B == 0 and Type == "other" and Shape == "chips":
    S_B = 0.6
if S_B == 0 and Type == "other" and Shape == "dust":
    S_B = 1.56
if S_B == 0 and Type == "other" and Shape == "fibres":
    S_B = 0.5
if S_B == 0 and Type == "other" and Shape == "particle":
    S_B = 1.16
if S_B == 0 and Type == "other" and Shape == "other":
    S_B = 1.2
if S_B == 0 and Type == "other":
    s_B= 0.88
#NAN Residence time
if Res_time == 0 and Type == "woody biomass" and Shape == "pellets":
    Res_time = 20
if Res_time == 0 and Type == "woody biomass" and Shape == "chips":
    Res_time = 80
if Res_time == 0 and Type == "woody biomass" and Shape == "dust":
    Res_time = 100
if Res_time == 0 and Type == "woody biomass" and Shape == "fibres":
    Res_time = 100
if Res_time == 0 and Type == "woody biomass" and Shape == "particle":
    Res_time = 40
if Res_time == 0 and Type == "woody biomass" and Shape == "other":
    Res_time = 80
if Res_time == 0 and Type == "woody biomass":
    Res_time=80
if Res_time == 0 and Type == "herbaceous biomass" and Shape == "pellets":
    Res_time = 20
if Res_time == 0 and Type == "herbaceous biomass" and Shape == "chips":
    Res_time = 80
if Res_time == 0 and Type == "herbaceous biomass" and Shape == "dust":
    Res_time = 100
if Res_time == 0 and Type == "herbaceous biomass" and Shape == "fibres":
    Res_time = 100
if Res_time == 0 and Type == "herbaceous biomass" and Shape == "particle":
    Res_time = 100
if Res_time == 0 and Type == "herbaceous biomass" and Shape == "other":
    Res_time =80
if Res_time == 0 and Type == "herbaceous biomass":
    Res_time=100
if Res_time == 0 and Type == "plastics" and Shape == "pellets":
    Res_time = 20
if Res_time == 0 and Type == "plastics" and Shape == "chips":
    Res_time = 80
if Res_time == 0 and Type == "plastics" and Shape == "dust":
    Res_time = 100
if Res_time == 0 and Type == "plastics" and Shape == "fibres":
    Res_time = 100
if Res_time == 0 and Type == "plastics" and Shape == "particle":
    Res_time = 40
if Res_time == 0 and Type == "plastics" and Shape == "other":
    Res_time = 80
if Res_time == 0 and Type == "plastics":
    Res_time=80
if Res_time == 0 and Type == "municipal solid waste" and Shape == "pellets":
    Res_time = 120
if Res_time == 0 and Type == "municipal solid waste" and Shape == "chips":
    Res_time = 80
if Res_time == 0 and Type == "municipal solid waste" and Shape == "dust":
    Res_time = 100
if Res_time == 0 and Type == "municipal solid waste" and Shape == "fibres":
    Res_time = 100
if Res_time == 0 and Type == "municipal solid waste" and Shape == "particle":
    Res_time = 80
if Res_time == 0 and Type == "municipal solid waste" and Shape == "other":
    Res_time = 80
if Res_time == 0 and Type == "municipal solid waste":
    Res_time= 90
if Res_time == 0 and Type == "sewage sludge" and Shape == "pellets":
    Res_time = 20
if Res_time == 0 and Type == "sewage sludge" and Shape == "chips":
    Res_time = 80
if Res_time == 0 and Type == "sewage sludge" and Shape == "dust":
    Res_time = 100
if Res_time == 0 and Type == "sewage sludge" and Shape == "fibres":
    Res_time = 100
if Res_time == 0 and Type == "sewage sludge" and Shape == "particle":
    Res_time = 255
if Res_time == 0 and Type == "sewage sludge" and Shape == "other":
    Res_time = 80
if Res_time == 0 and Type == "sewage sludge":
    Res_time= 90
if Res_time == 0 and Type == "other" and Shape == "pellets":
    Res_time = 40
if Res_time == 0 and Type == "other" and Shape == "chips":
    Res_time = 80
if Res_time == 0 and Type == "other" and Shape == "dust":
    Res_time = 100
if Res_time == 0 and Type == "other" and Shape == "fibres":
    Res_time = 100
if Res_time == 0 and Type == "other" and Shape == "particle":
    Res_time = 40
if Res_time == 0 and Type == "other" and Shape == "other":
    Res_time = 80
if Res_time == 0 and Type == "other":
    Res_time= 80

#NAN conditon CHONS LHV ASH MOISt Volatile fix-carbon ER Cell Hemicell Lignin Temp Res_time S/B
if LHV==0:
    LHV=18.18
if Ash==0:
    Ash=2.05
if Moisture==0:
    Moisture=7.65
if Volatile==0:
    Volatile=80.98
if Fixed_carbon==0:
    Fixed_carbon=14.46
if ER==0:
    ER=0.3
if Cellulose==0:
    Cellulose=0
if Hemicellulose==0:
    Hemicellulose=0
if Lignin==0:
    Lignin=0
if Temp==0:
    Temp=800
if S_B==0:
    S_B=0.8

#NAN conditon categories
if Scale=="other":
    Scale='lab'
if Operating_con=="other":
    Operating_con='continuous'
if Operating_p=="other":
    Operating_p='atmospheric'
if reactor=="other bed":
    reactor='air'
if bed=="other":
    bed='silica'
#cat
cat=0
if cat=="Yes":
    cat=1
elif cat=="No":
     cat=0
#type
woody_biomass=0
sewage_sludge=0
herbaceous=0
plastics=0
MSW=0
other_type=0
if Type == "woody biomass":
    woody_biomass = 1
elif Type == "herbaceous biomass":
    herbaceous =1
elif Type == "sewage sludge":
    sewage_sludge =1
elif Type == "plastics":
    plastics =1
elif Type == "municipal solid waste":
    MSW =1
elif Type == "other":
    other_type =1
#shape
chips=0
dust=0
fibres=0
particle=0
pellet=0
other_shape=0
if Shape == "chips":
    chips = 1
elif Shape == "dust":
    dust =1
elif Shape == "fibres":
    fibres =1
elif Shape == "particle":
    particle =1
elif Shape == "pellets":
    pellet =1
elif Shape == "other":
    other_shape =1

#Scale
pilot=0
lab=0
if Scale == "pilot":
    pilot = 1
elif Scale == "lab":
    lab =1

#operating conditon
batch=0
continuous=0
if Operating_con == "batch":
    batch = 1
elif Operating_con == "continuous":
    continuous =1

#operating pressure
num_pressure=0
num_pressure2=0
atmospheric=0
slightly_a=0
slightly_b=0
if Operating_p == "205":
    num_pressure = 1
elif Operating_p == "atmospheric":
    atmospheric =1
elif Operating_p == "slightly above atmospheric":
    slightly_a =1
elif Operating_p == "slightly below atmospheric":
    slightly_b =1
elif Operating_p == "111.46":
    num_pressure2 =1

#agent
air=0
steam_air=0
other_gasify=0
oxygen=0
steam_oxygen=0
steam=0
if gasifying == "air":
    air = 1
elif gasifying == "air+steam":
    steam_air =1
elif gasifying == "other":
    other_gasify =1
elif gasifying == "oxygen":
    oxygen =1
elif gasifying == "steam+oxygen":
    steam_oxygen =1
elif gasifying == "steam":
    steam =1

#reactor type
fix_bed=0
flu_bed=0
horizontal=0
other_bed=0
if reactor == 'fixed bed':
    fix_bed = 1
elif reactor == 'fluidised bed':
    flu_bed =1
elif reactor == 'horizontal tube':
    horizontal =1
elif reactor == 'other bed':
    other_bed =1
#bed
alumina=0
Y_alumina=0
Calcium_oxide=0
dolomite=0
olivine=0
silica=0
sand=0
if bed == 'alumina':
    alumina = 1
elif bed == 'Y-alumina':
    Y_alumina =1
elif bed == 'calcium oxide':
    Calcium_oxide =1
elif bed == 'dolomite':
    dolomite =1
elif bed == 'olivine':
    olivine =1
elif bed == 'silica' :
    silica =1
elif bed == 'sand':
    sand =1

input_data = [Particle_size, LHV, Ash, Moisture,Volatile,Carbon,Hydrogen,Oxygen,Nitrogen,Sulfur,Fixed_carbon,ER,Cellulose,
              Hemicellulose,Lignin,Temp,Res_time,S_B,cat,
              woody_biomass,herbaceous,sewage_sludge,plastics,MSW,other_type,
              chips,dust,fibres,particle,pellet,other_shape,
              pilot,lab,
              batch,continuous,
              num_pressure,atmospheric,slightly_a,slightly_b,num_pressure2,
              air,steam_air,other_gasify,oxygen,steam_oxygen,steam,
              fix_bed,flu_bed,horizontal,other_bed,
              alumina,Calcium_oxide,Y_alumina,dolomite,olivine,silica,sand]
data = [input_data]
columns=['feed_particle_size','feed_LHV','feed_ash','feed_moisture','feed_VM','C','H','O','N','S','feed_FC','ER','feed_cellulose',
         'feed_hemicellulose','feed_lignin','temperature','residence_time','steam_biomass_ratio','catalyst'
         ,"Swoody biomass","herbaceous biomass","sewage sludge","plastics","municipal solid wast","other_feed_type",
         "chips","dust","fibres","particles","pellets","other_feed_shape",
         "pilot","lab",
         "batch","continuous",
         "205","atmospheric","slightly above atmospheric","slightly below atmospheric","111.46",
         "air","air + steam","other_gas","oxygen","steam+oxygen","steam",
         'fixed bed','fluidised bed','horizontal ','other_bed',
         'alumina','calcium oxide','Y-alumina','dolomite','olivine','silica','sand']
df = pd.DataFrame(data, columns=columns)
#จัดcolumn
desired_order = ['feed_particle_size','feed_LHV','C','H','N','S','O','feed_ash','feed_moisture','feed_VM',
                 'feed_FC','feed_cellulose','feed_hemicellulose','feed_lignin','temperature','residence_time',
                 'steam_biomass_ratio','ER','catalyst','herbaceous biomass','municipal solid wast',
                 'other_feed_type','plastics','sewage sludge','Swoody biomass','chips','dust','fibres',
                 'other_feed_shape','particles','pellets','batch','continuous','111.46','205','atmospheric',
                 'slightly above atmospheric','slightly below atmospheric','air','air + steam','other_gas','oxygen',
                 'steam','steam+oxygen','horizontal ', 'fixed bed','fluidised bed','other_bed','alumina','Y-alumina'
                 ,'calcium oxide','dolomite','olivine','silica','sand','lab','pilot']
desired_num=['feed_particle_size','feed_LHV','C','H','N','S','O','feed_ash','feed_moisture','feed_VM',
                 'feed_FC','feed_cellulose','feed_hemicellulose','feed_lignin','temperature','residence_time',
                 'steam_biomass_ratio','ER','catalyst']
desired_Cat=['herbaceous biomass','municipal solid wast',
                 'other_feed_type','plastics','sewage sludge','Swoody biomass','chips','dust','fibres',
                 'other_feed_shape','particles','pellets','batch','continuous','111.46','205','atmospheric',
                 'slightly above atmospheric','slightly below atmospheric','air','air + steam','other_gas','oxygen',
                 'steam','steam+oxygen','horizontal ', 'fixed bed','fluidised bed','other_bed','alumina','Y-alumina'
                 ,'calcium oxide','dolomite',	'olivine',	'silica',	'sand',	'lab',	'pilot']
data_model_ordered = df[desired_order]
categories=df[desired_Cat]
number=df[desired_num]
data_model = pd.read_excel('max_min.xlsx')
x_model = data_model.loc[:,~data_model.columns.isin(
            ['N2', 'H2', 'CO', 'CO2', 'CH4', 'C2Hn', 'gas_LHV', 'gas_tar', 'gas_yield', 'char_yield', 'CGE',
             'CCE','herbaceous biomass','municipal solid wast',
                 'other_feed_type','plastics','sewage sludge','Swoody biomass','chips','dust','fibres',
                 'other_feed_shape','particles','pellets','batch','continuous','111.46','205','atmospheric',
                 'slightly above atmospheric','slightly below atmospheric','air','air + steam','other_gas','oxygen',
                 'steam','steam+oxygen','horizontal ', 'fixed bed','fluidised bed','other_bed','alumina','Y-alumina'
                 ,'calcium oxide','dolomite',	'olivine',	'silica',	'sand',	'lab',	'pilot'])]

min_values = x_model.min()
max_values = x_model.max()

normalized_unknown= ((number - min_values) /( max_values -  min_values))
normalize_input = normalized_unknown[desired_num]

result =pd.concat([normalize_input,categories],axis=1)


unknown=result[['feed_particle_size','C','H','S','feed_ash','feed_moisture',
                 'temperature',
                 'steam_biomass_ratio','ER','catalyst','herbaceous biomass','municipal solid wast',
                 'plastics','sewage sludge','Swoody biomass','chips','dust','fibres',
                 'particles','pellets','batch',
                 'air','air + steam','oxygen',
                 'steam','steam+oxygen','horizontal ', 'fixed bed','fluidised bed','alumina','Y-alumina'
                 ,'calcium oxide',	'olivine',	'silica',	'sand',	'lab',	'pilot']]

clf = joblib.load('Gfinalized_model.joblib')
data = pd.read_excel('Prepared_Datasets.xlsx')
x = data.loc[:, ~data.columns.isin(
    ['N2', 'H2', 'CO', 'CO2', 'CH4', 'C2Hn', 'gas_LHV', 'gas_tar', 'gas_yield', 'char_yield', 'CGE',
     'CCE', 'feed_cellulose', 'feed_hemicellulose', 'feed_lignin', 'residence_time', '111.46',
     '205', 'atmospheric', 'slightly above atmospheric', 'slightly below atmospheric',
     'feed_LHV', 'feed_VM', 'feed_FC', 'other_feed_type', 'other_feed_shape',
     'calcium oxide','continuous', 'other_gas', 'other_bed', 'dolomite','N','O'])]

button_style = """
    <style>
        div.stButton > button {
            color: black;
            background-color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 32px;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)
st.subheader("Hydrogen Prediction")
st.markdown(":point_down: <span style='color:Maroon'>***Click below!!!***",unsafe_allow_html=True)
if st.button(':globe_with_meridians: Predict Hydrogen product'):
    prediction = clf.predict(unknown)
    Ans=(prediction*(71.5-3.1)+3.1)
    st.subheader("% Hydrogen :")
    st.text(Ans)
st.write(":warning: WARNING  :warning:")
st.caption('The system will automatically fill in the missing values with the median value of the data in the database if any data is missing or incomplete throughout the data filling process.')
if st.button("More info"):
    st.caption("The table illustrates the data-filling process for data types that are unrelated to each other or where the relationship between the data cannot be determined.")
    normal = pd.read_excel('filling_data_normal.xlsx')
    st.write(normal)
    st.caption("The table demonstrates the data filling process for data types that are interrelated.")
    complex = pd.read_excel('filling_data_complex.xlsx')
    st.write(complex)
    st.button("Back", type="primary")

st.divider()
#--------------------------------------DATA BASE--------------------------------------------------------#
st.header("	:round_pushpin: Experimental Data For Training The Model (collected from several literature) :speech_balloon:",divider='red')
#-----------------------------------------------------------------------------------------------------#

file = "2023_Data_BiomassGasification_NED.xlsx"
df = load_data()
df['Operating pressure'] = df['Operating pressure'].astype(str)
create_data = {
                "Feed types": "multiselect",
                "Feed shapes": "multiselect",
               "Operating condition": "multiselect",
                "Operating pressure": "multiselect",
               'Gasifying agent':"multiselect",
               'Reactor type':"multiselect",
               'Bed material':"multiselect",
               'Scale':"multiselect"}
with st.sidebar:
    st.title("Data Filtering :mag:")
    st.caption("(Use to select/find the desired data range)")
all_widgets = sp.create_widgets(df, create_data, ignore_columns=["ID",'Reference'])
res = sp.filter_df(df, all_widgets)
#DATA########
st.write(res)
st.markdown(":point_left: <span style='color:Maroon'>***You can select the desired categoricals from the options on the left.***",unsafe_allow_html=True)
if st.button("Assumption info"):
    st.caption("-- Assumption --")
    st.caption(
        "(i)Parameters were converted to the basis used in the data set (e.g. ultimate composition data given as wet basis (wb) would be converted to dry ash-free basis (daf) to fit in with the rest of the data set).")
    st.caption("(ii)The feedstock's LHV was calculated from the feedstocks HHV where necessary.")
    st.caption("(iii)The quoted particle size refers to the lowest dimension of the particle ")
    st.caption("(iv)All higher order hydrocarbons (C2Hn) in the Syngas were treated as C2H4. ")
    st.caption(
        "(v)Cold gas efficiency (CGE) and carbon conversion efficiency (CCE) were calculated where necessary. Hence some efficiencies >100% were obtained.  ")
    st.button("Back", type="primary")
#download CSV
csv = convert_df(df)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='data_df.csv',
    mime='text/csv')

st.divider()
#-------------------------------------------------------------------------------------------------------------#
#picture########
st.header("	:round_pushpin: Result : :loudspeaker: ")
st.info(':bulb: Part I : Exploratory Data Analysis (EDA)')
st.write("- EDA assists you in gaining insights into data, identifying trends, and discovering any patterns "
         "or anomalies.")

tab1, tab2= st.tabs(["Pearson", "Boxplot"])
with tab1:  # Pearson
        st.subheader("Pearson Correlation")
        col1, col2 = st.columns([2, 1])
        # Code Pearson
        data = pd.read_excel('Prepared_Datasets.xlsx')
        input = data.loc[:, data.columns.isin(
            ['feed_particle_size', 'feed_LHV', 'C', 'H', 'O', 'N', 'S', 'temperature', 'steam_biomass_ratio', 'ER',
             'feed_VM', 'feed_FC', 'feed_ash', 'feed_moisture'])]

        pearson = []
        for i in range(len(input.columns)):
            for j in range(len(input.columns)):
                x = input[input.columns[i]].values
                y = input[input.columns[j]].values
                corr, _ = pearsonr(x, y)
                pearson.append(corr)

        pearson = np.array(pearson).reshape(14, 14)
        axis = ['PS', 'LHV', 'C', 'H', 'N', 'S', 'O', 'ASH', 'M', 'VM', 'FC', 'T', 'SB', 'ER']
        df = pd.DataFrame(data=pearson,
                          index=axis,
                          columns=axis)
        fig, ax = plt.subplots()
        bar = sns.heatmap(df.iloc[:, :14], cbar=1, linewidths=1, vmax=1, vmin=-1, center=0, square=True, cmap='YlGnBu',
                          fmt='.2f', annot=True, cbar_kws={'shrink': 1}, annot_kws={"size": 7})

        ax.scatter(axis, axis)
        col1.pyplot(fig)
        with col2:
            st.write(" - Pearson correlation is a statistical measure that quantifies the strength and direction of a linear relationship between two continuous variables.")
            st.write("- Pearson correlation is employed to eliminate variables with low or excessive intercorrelation, aiming to reduce the complexity of the model's operation.")
            st.write(" - This model discards the variables with the correlation coefficients exceeding 0.6 (LHV ,N ,O ,Fixed carbon, Volatile matter). ")

with tab2:  # Boxplot
        st.subheader('Boxplot')
        st.write(' - The box plot below shows the distribution of each input variables.')
        data = pd.read_excel('2023_Data_BiomassGasification_NED.xlsx', 'treated')
        input_box = data.loc[:, data.columns.isin(
            ['feed_particle_size', 'feed_LHV', 'C', 'H', 'O', 'N', 'S', 'temperature', 'steam_biomass_ratio', 'ER',
             'feed_VM', 'feed_FC', 'feed_ash', 'feed_moisture'])]
        df = pd.DataFrame(input_box)
        box_plot = st.selectbox('select', df.columns)

        if box_plot:
            fig2, ax = plt.subplots()
            ax.set_xlabel(box_plot)
            sns.histplot(df[box_plot], bins=100, kde=True, color='red', edgecolor='black', cumulative=False,
                         linewidth=1.2, multiple="layer")
            st.pyplot(fig2)

        st.subheader("Units")
        st.markdown("- Feed particle size (mm)")
        st.markdown("- Heating value (MJ/Nm^3)")
        st.markdown("- C,H,O,N,S (wt.% dry basis)")
        st.markdown("- Ash, Moisture, Volatile matter, Fixed carbon (wt.% dry basis)")
        st.markdown("- Temperature (C)")

st.info(':bulb: Part II : Selected Predictive Model & Modeling Evaluation')
st.write("- Displaying experimental results, data relationships, and prediction model performance, including the reliability of the model.")

tab3,tab4= st.tabs(["Model","SHAP"])
with tab4:
        st.subheader('SHapley Additive exPlanations')
        model=joblib.load('Gfinalized_model.joblib')
        data = pd.read_excel('Prepared_Datasets.xlsx')

        x = data.loc[:, ~data.columns.isin(
         ['N2', 'H2', 'CO', 'CO2', 'CH4', 'C2Hn', 'gas_LHV', 'gas_tar', 'gas_yield', 'char_yield', 'CGE',
         'CCE', 'feed_cellulose', 'feed_hemicellulose', 'feed_lignin', 'residence_time', '111.46',
         '205', 'atmospheric', 'slightly above atmospheric', 'slightly below atmospheric',
         'feed_LHV', 'feed_VM', 'feed_FC', 'other_feed_type', 'other_feed_shape',
         'continuous', 'other_gas', 'other_bed', 'dolomite','N','O'])]
        y = data['H2']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

        explainer = shap.Explainer(model)
        shap_values = explainer(x_test)
        st.write("- The SHAP values assign the relative relevance of each feature to the model's output,"
                 " which helps to explain the model's prediction. They provide information about each"
                 " feature's relative contribution to the final forecast for a specific situation.")
        plt.figure()
        st_shap(shap.plots.beeswarm(shap_values), height=300)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], x_test.iloc[0, :]), height=200, width=1000)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000, :], x_test.iloc[:1000, :]), height=400,width=1000)
with tab3:
        col5,col6=st.columns(2)
        with col5:
            st.subheader('Model Evaluation')
            model_GB = joblib.load('Gfinalized_model.joblib')
            data = pd.read_excel('Prepared_Datasets.xlsx')
            x = data.loc[:, ~data.columns.isin(
            ['N2', 'H2', 'CO', 'CO2', 'CH4', 'C2Hn', 'gas_LHV', 'gas_tar', 'gas_yield', 'char_yield', 'CGE',
             'CCE', 'feed_cellulose', 'feed_hemicellulose', 'feed_lignin', 'residence_time', '111.46',
             '205', 'atmospheric', 'slightly above atmospheric', 'slightly below atmospheric',
             'feed_LHV', 'feed_VM', 'feed_FC', 'other_feed_type', 'other_feed_shape',
             'continuous', 'other_gas', 'other_bed', 'dolomite','N','O'])]
            y = data['H2']
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
            Y = y_test
            Y.reset_index(drop=True, inplace=True)
            y_pred = model.predict(x_test)
            df_pred = pd.DataFrame({'H2': y_pred.flatten()})
            Yt = y_train
            Yt.reset_index(drop=True, inplace=True)
            y_ptrain = model.predict(x_train)
            df_train = pd.DataFrame({'H2': y_ptrain.flatten()})
            fig, ax = plt.subplots()
            plt.plot([0, 1], [0, 1], label='identity', color='black', linestyle='-')
            plt.scatter(df_train['H2'], Yt, label='train', color='crimson')
            plt.scatter(df_pred['H2'], Y, label='test', color='royalblue')
            ax.set_title('Actual vs. Predicted Values')
            ax.set_xlabel('Predictions')
            ax.set_ylabel('Actual')
            ax.legend()
            st.pyplot(fig)
            st.markdown("- Gradient Boosting Regressor with no cross-validation")
            st.caption('- [ The model obtained is the most accurate, as determined by comparing the R-squared values of '
                       'the Extra tree, Gradient Boosting Regressor, Multi-layer Perceptron, and K-Nearest Neighbor '
                       'models with both cross-validation and without cross-validation. ]')
        with col6:
            def performance_evaluation(actual, predict):  # Model evaluation function

                # Computing R2 Score
                r2 = r2_score(actual, predict)

                # Computing Mean Square Error (MSE)
                mse = mean_squared_error(actual, predict)

                # Computing Mean Absolute Error (MAE)
                mae = mean_absolute_error(actual, predict)
                st.markdown(f"> - R-squared : <span style='color:red'>{r2:.{5}f}", unsafe_allow_html=True)
                st.markdown(f"> - Mean Square Error : <span style='color:red'>{mse:.{5}f}",unsafe_allow_html=True)
                st.markdown(f"> - Mean Absolute Error : <span style='color:red'>{mae:.{5}f}",unsafe_allow_html=True)
                st.divider()
                eval_list = [r2, mse, mae]

                return eval_list
            st.write('<u style="font-size: 24px;">Training set</u>', unsafe_allow_html=True)
            eval_train = performance_evaluation(Yt, df_train["H2"])
            st.write('<u style="font-size: 24px;">Test set</u>', unsafe_allow_html=True)
            eval_test = performance_evaluation(Y, df_pred["H2"])
#-----------------------------------------------------------------------------------------------------#
