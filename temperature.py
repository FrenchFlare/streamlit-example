#IMPORTATION MODULES NECESSAIRES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
install streamlit
import streamlit as st
import plotly.express as px

st.title("Projet température terrestre")
st.sidebar.title("Sommaire")
pages = ["Intro", "carte monde", "entrainement"," feature importance"]
page = st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write("### Introduction")
    st.image("CO2.jpg")
    st.markdown("Impact de l'Homme sur la temperature terrestre")

#CHARGEMENT DATA CO2
em_co2 = pd.read_csv('C:/Users/Adrien/Documents/PROJET/owid-co2-data.csv', sep = ',')
CO2 = em_co2.groupby(['country','year']).sum().reset_index()
CO2 = CO2.sort_values(['year'])
CO2['year'] = CO2['year'].astype('int')
CO2 = CO2.rename(columns={'year':'Year'})

#CHARGEMENT DATA TEMPERATURE
temperature = pd.read_csv('C:/Users/Adrien/Documents/PROJET/GLB.Ts+dSST.csv',
                          skiprows = 1, sep = ',')

#PREPARATION DES DONNEES : REMPLACEMENT, SUPPRESSION, MODIFICATION
temperature = temperature.replace('*******', np.NaN)
temperature.dropna(axis =0, inplace =True)
temp = temperature[['Year','J-D']]
temp = temp.groupby(['Year','J-D']).sum().reset_index()
temp = temp.sort_values(['Year','J-D'])
temp = temp.rename(columns={'J-D':'temp'})
temp['temp'] = temp['temp'].str.replace(' ', '')
temp = temp[:-1]
temp = temp.astype(float)
temp['Year'] = temp['Year'].astype(int)

#creation fichier mixte CO2-TEMP
mixed = pd.merge(temp,CO2, how = 'left', left_on =['Year'], right_on = ['Year'] )

filtered = mixed[['Year','temp','country','co2_including_luc','co2','consumption_co2','nitrous_oxide','oil_co2','total_ghg','methane','gdp']]
World = filtered[filtered['country'].isin(['World'])]


#RECHERCHE CORRELATIONS ENTRE TEMPERATURE ET CO2
#PREPARATION DONNEES ENTRAINEMENT ET TEST
from sklearn.model_selection import train_test_split
feats = World.drop(['Year', 'temp','country'], axis =1)
target = World['temp']
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state = 42)

#LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
predictions_reg = reg.predict(X_test)
erreurs_reg = predictions_reg - y_test

#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth = 3, random_state=42)
tree.fit(X_train, y_train)
predictions_tree = tree.predict(X_test)
erreurs_tree = predictions_tree - y_test

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
predictions_forest = forest.predict(X_test)
erreurs_forest = predictions_forest - y_test

#SupportVectorRegression
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
print('Score SVR sur ensemble train', svr.score(X_train, y_train))
print('Score SVR sur ensemble test', svr.score(X_test, y_test))
print('nous pouvons éliminer le SVR')

#GRADIENTBOOSTINGREGRESSOR
from sklearn.ensemble import GradientBoostingRegressor
gb_reg=GradientBoostingRegressor()
gb_reg.fit(X_train,y_train)
predictions_gb_reg = gb_reg.predict(X_test)
erreurs_gb_reg = predictions_gb_reg - y_test

#travail sur MSE RMSE et MAE
def metrics(errors):
    import numpy as np
    mse = (errors**2).mean()
    rmse = np.sqrt((errors**2).mean())
    mae = np.abs(errors).mean()
    return mse, rmse, mae

if page == pages[1]:
    st.write("carte du monde des émissions de CO2")
    if st.checkbox("Afficher la carte du monde"):
        fig = px.choropleth(CO2, locations="country",
                    color=np.log10(CO2["co2_including_luc"]),
                    hover_name="Year",
                    hover_data=["co2_including_luc"],
                    locationmode="country names",
                    animation_frame='Year',
                    color_continuous_midpoint = 3,
        color_continuous_scale=px.colors.sequential.thermal_r)
        fig.update_layout(margin=dict(l=20,r=0,b=0,t=70,pad=0),paper_bgcolor="white",height= 700,title_text = 'CO2 usage',font_size=18)
        fig.show()

if page == pages[2]:
    st.write('# ENTRAINEMENT DE MODELES')
    st.write('## REGRESSION LINEAIRE :')
    st.write(' MSE, RMSE, MAE :',metrics(erreurs_reg))
    st.write('Score reg sur ensemble train',reg.score(X_train, y_train))
    st.write('Score reg sur ensemble test', reg.score(X_test, y_test))
    st.write('## ARBRE DE DECISION :')
    st.write(' MSE, RMSE, MAE :',metrics(erreurs_forest))
    st.write('Score tree sur ensemble train', tree.score(X_train, y_train))
    st.write('Score tree sur ensemble test', tree.score(X_test, y_test))
    st.write('## RANDOM FOREST :')
    st.write(' MSE, RMSE, MAE :',metrics(erreurs_tree))
    st.write('Score RandomForest sur ensemble train', forest.score(X_train, y_train))
    st.write('Score RandomForest sur ensemble test', forest.score(X_test, y_test))
    st.write('## GRADIENT BOOSTING :')
    st.write('MSE, RMSE, MAE :',metrics(erreurs_gb_reg))
    st.write('Score GradientBoostingRegressor sur ensemble train', gb_reg.score(X_train, y_train))
    st.write('Score GradientBoostingRegressor sur ensemble test', gb_reg.score(X_test, y_test))
    st.write("les meilleurs modeles sont donc l'arbre de décision, le Gradient et le Random Forest.")
    st.write("Nous allons observer les features les plus importants dans la page suivante")


#affichage importance variables
#GRADIENT

feat_importances1 = pd.DataFrame({
    "Variables":
    feats.columns,
    "Importance":
    gb_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)



#RANDOM FOREST
feat_importances2 = pd.DataFrame({
    "Variables":
    feats.columns,
    "Importance":
    forest.feature_importances_
}).sort_values(by='Importance', ascending=False)


feat_importances2.nlargest(8, "Importance").plot.bar(x="Variables",
                                                    y="Importance",
                                                    figsize=(15, 5),
                                                    color="#4529de");
plt.title('Forest importance')

#ARBRE DE DECISION
feat_importances3 = pd.DataFrame({
    "Variables":
    feats.columns,
    "Importance":
    tree.feature_importances_
}).sort_values(by='Importance', ascending=False)
feat_importances3.nlargest(8, "Importance").plot.bar(x="Variables",
                                                    y="Importance",
                                                    figsize=(15, 5),
                                                    color="#4529de");
plt.title('tree importance')

if page == pages[3]:    
    st.write("# IMPORTANCE DES VARIABLES :")
    st.write('## GRADIENT')
    st.bar_chart(feat_importances1, x = "Variables", y = "Importance")
    st.write('## FOREST')
    st.bar_chart(feat_importances2, x = "Variables", y = "Importance")
    st.write('## TREE')
    st.bar_chart(feat_importances3, x = "Variables", y = "Importance")
    
    
    
    
    
    
    