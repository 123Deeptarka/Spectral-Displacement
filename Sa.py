# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:41:02 2024

@author: deeptarka.roy
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor


st.markdown('<h1 style="font-size: 40px; font-weight: bold;"> Displacement Demands </h1>', unsafe_allow_html=True)

st.sidebar.header("Specify Input Parameters")


#import data

df = pd.read_excel('A.xlsx')
x = df[["D","LD","fc","fyl","fyt","pl","pt","Ny"]]




y = df[["DS1","DS2","DS3","DS4","F1","F2","F3","F4"]]


model=RandomForestRegressor()
#model=RandomForestRegressor()
model.fit(x,y)

def user_input_features():
    D=st.sidebar.number_input("D",x.D.min(),x.D.max(),x.D.mean())
    LD =st.sidebar.number_input("LD",x.LD.min(),x.LD.max(),x.LD.mean())   
    fc =st.sidebar.number_input("fc",x.fc.min(),x.fc.max(),x.fc.mean())
    fyl =st.sidebar.number_input("fyl",x.fyl.min(),x.fyl.max(),x.fyl.mean())
    fyt =st.sidebar.number_input("fyt",x.fyt.min(),x.fyt.max(),x.fyt.mean())
    pl =st.sidebar.number_input("pl",x.pl.min(),x.pl.max(),x.pl.mean())
    pt =st.sidebar.number_input("pt",x.pt.min(),x.pt.max(),x.pt.mean())
    Ny =st.sidebar.number_input("Ny",x.Ny.min(),x.Ny.max(),x.Ny.mean())
    data={"D":D,"LD":LD,"fc":fc,"fyl":fyl,"fyt":fyt,"pl":pl,"pt":pt,"Ny":Ny}
    features=pd.DataFrame(data,index=[0])
    features_round=features.round(2)
    return features_round

Data=user_input_features()
prediction=model.predict(Data)

#Calculate Time Period of Structure 

I=(3.14*(Data["D"])**4)/64
K=3*200000*I/(Data["D"]*Data["LD"])**3
P=(Data["Ny"]*Data["fc"]*3.14*(Data["D"]**2))/4000
T=2*3.14*(0.1*P/K)**0.5


st.markdown('<h1 style="font-size: 40px; font-weight: bold;"> Seismic Response Spectrum </h1>', unsafe_allow_html=True)


# Predefined x values (time periods)
x_data = [0.2, 0.5, 1, 2, 5, 10]

# Create input fields for Sa values
y_data_2 = []
y_data_5= []
y_data_10=[]
for i in range (6):
    y_2 = st.sidebar.number_input(f"Sa at 2%/50 yrs  value for time period {x_data[i]} sec:", min_value=0.0, format="%.2f")
    y_5 = st.sidebar.number_input(f"Sa at 5%/50 yrs  value for time period {x_data[i]} sec:", min_value=0.0, format="%.2f")
    y_10 = st.sidebar.number_input(f"Sa at 10%/50 yrs  value for time period {x_data[i]} sec:", min_value=0.0, format="%.2f")
    y_data_2.append(y_2)
    y_data_5.append(y_5)
    y_data_10.append(y_10)
    
    
# Convert lists to numpy arrays
x_data = np.array(x_data)
y_data_2 = np.array(y_data_2)
y_data_5 = np.array(y_data_5)
y_data_10 = np.array(y_data_10)


# Create plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_data, y_data_2, "o--",label="2%/50 yrs")
ax.plot(x_data, y_data_5, "o--",label="5%/50 yrs")
ax.plot(x_data, y_data_10, "o--",label="10%/50 yrs")
ax.set_xlabel('Period (s)')
ax.set_ylabel('Spectral Acceleration, Sa (g)')
#ax.set_title('Response Spectrum')
ax.legend()
#ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)

# Optional: Display the input data
#st.write("Input Data:")
#st.write({"Period (s)": x_data, "Sa (g)": y_data_2})

# Create interpolation function
f_2 = interpolate.interp1d(x_data, y_data_2, kind='linear')
f_5 = interpolate.interp1d(x_data, y_data_5, kind='linear')
f_10 = interpolate.interp1d(x_data, y_data_10, kind='linear')

# Create a finer x array for smooth interpolation
#x_interp = np.linspace(x_data.min(), x_data.max(), 100)

# Calculate interpolated y values
#y_interp = f(x_interp)
Sa_2=f_2(T)
Sa_5=f_5(T)
Sa_10=f_10(T)
# Plot original data and interpolated curve
plt.figure(figsize=(6, 3))
plt.plot(x_data, y_data_2, "o--")

#plt.plot(x_interp, y_interp, color='blue', label='Interpolated curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('User Input Data with Linear Interpolation')
#plt.legend()
plt.grid(True)
plt.show()


print(Sa_2)
Sd_2=(Sa_2*(T)**2)/(4*3.14**2)*10000
Sd_2_=Sd_2*100/(Data["D"]*Data["LD"])
print(Sd_2_)

Sd_5=(Sa_5*(T)**2)/(4*3.14**2)*10000
Sd_5_=Sd_5*100/(Data["D"]*Data["LD"])
print(Sd_5_)

Sd_10=(Sa_10*(T)**2)/(4*3.14**2)*10000
Sd_10_=Sd_10*100/(Data["D"]*Data["LD"])
print(Sd_10_)

#st.header("Predicted Damage States ")

P=pd.DataFrame(prediction,columns=["DS1","DS2","DS3","DS4","F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"])

P_DS=P[["DS1","DS2","DS3","DS4"]]


#st.dataframe(P_DS,hide_index=True)
P_F=P[["F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"]]
#P_DS_display = P_DS.reset_index(drop=True)
#st.write(P_DS)
#styles = [
 #   dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
 #   dict(selector="td", props=[("font-size", "16px"),("font-weight", "bold") ,("color", "#484848")])
#]
# Apply styling to dataframe
#styled_df = P_DS.style.set_table_styles(styles)
#st.table(styled_df)

# Title
#st.write("Drift Ratio")

# Title with Markdown for styling
#st.markdown("<h1 style='text-align: center; font-size: 20px; font-weight: bold; color: #484848;'>Drift Ratio (%)</h1>", unsafe_allow_html=True)

# Subtitles
#st.write("DS1, DS2, DS3, DS4")

# Styling
#styles = [
   # dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
   # dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold")    ,("color", "#484848")])
#]
styles = [
    dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="table", props=[("border-collapse", "collapse")])  # Ensures borders are collapsed
]

# Apply styling to dataframe
styled_df = P_DS.style.set_table_styles(styles).format("{:.2f}").hide(axis="index")

#html = styled_df.to_html(index=False)
#st.write(html,unsafe_allow_html=True)
# Display the table
#st.table(styled_df)
P_DS_no_index = P_DS.round(2).reset_index(drop=True)
html = P_DS_no_index.to_html(index=False)

# Apply custom CSS to the HTML table
html = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 3px solid black !important;  /* Thicker border with !important */
        padding: 4px;
        text-align: center;
    }}
    th {{
        font-size: 20px;
        font-weight: bold;
        color: #484848;
        border:4px solid #484848;
    }}
    td {{
        font-size: 16px;
        font-weight: bold;
        color: #484848;
        border:4px solid #484848;
    }}
</style>
{html}
"""

# Display the HTML table
#st.markdown(html, unsafe_allow_html=True)






Outpred_DR_0=P_DS.to_numpy().reshape(4,)

Outpred_F_0=P_F.to_numpy().reshape(4,)
C=[Sd_2_]
c=np.array(C)
c_1D=c.flatten()
D=[Sd_5_]
d=np.array(D)
d_1D=d.flatten()
E=[Sd_10_]
e=np.array(E)
e_1D=e.flatten()
print(c_1D.shape)

S = pd.DataFrame({
    'Sd_2': c_1D,
    'Sd_5': d_1D,
    'Sd_10': e_1D
})


# Title with Markdown for styling
st.markdown("<h1 style='text-align: center; font-size: 20px; font-weight: bold; color: #484848;'>Spectral Displacement (%)</h1>", unsafe_allow_html=True)

# Subtitles
#st.write("DS1, DS2, DS3, DS4")

# Styling
#styles = [
   # dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
   # dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold")    ,("color", "#484848")])
#]
stylesnew = [
    dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="table", props=[("border-collapse", "collapse")])  # Ensures borders are collapsed
]

# Apply styling to dataframe
#styled_df = df.style.set_table_styles(stylesnew).format("{:.2f}").hide(axis="index")

#html = styled_df.to_html(index=False)
#st.write(html,unsafe_allow_html=True)
# Display the table
#st.table(styled_df)
S_no_index = S.round(2).reset_index(drop=True)
html = S_no_index.to_html(index=False)

# Apply custom CSS to the HTML table
html = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 3px solid black !important;  /* Thicker border with !important */
        padding: 4px;
        text-align: center;
    }}
    th {{
        font-size: 20px;
        font-weight: bold;
        color: #484848;
        border:4px solid #484848;
    }}
    td {{
        font-size: 16px;
        font-weight: bold;
        color: #484848;
        border:4px solid #484848;
    }}
</style>
{html}
"""

# Display the HTML table
st.markdown(html, unsafe_allow_html=True)





# In[20]:


a=np.insert(Outpred_DR_0,0,0)
b=np.insert(Outpred_F_0,0,0)

# In[21]:
st.header("Predicted Pushover Curve")
fig,ax=plt.subplots(figsize=(6,3))
ax.plot(a,b,label="Predicted Pushover Curve",marker="o")
plt.axvline(x=c, color='red', linestyle='--',label="2%/50 yrs")
plt.axvline(x=d, color='blue', linestyle='--',label="5%/50 yrs ")
plt.axvline(x=e, color='green', linestyle='--',label="10%/50 yrs")


#ax.plot(a1,b1,label="Simulated Pushover Curve",marker="o")
ax.set_xlabel("Drift Ratio (%)")
ax.set_ylabel("Force (kN)")
#ax.set_title("Predicted VS Simulated Pushover Curves")
ax.legend()




#ax.show()
#st.pyplot(fig)
# Label the points
for i in range(1, 5):  # We start from 1 to skip the (0,0) point
    ax.annotate(f'DS{i}', (a[i], b[i]), textcoords="offset points", xytext=(5,-20), ha='center')

# Add grid for better readability
#ax.grid(True, linestyle='--', alpha=0.7)

# Use tight layout to prevent clipping of labels
plt.tight_layout()

st.pyplot(fig, use_container_width=True)
