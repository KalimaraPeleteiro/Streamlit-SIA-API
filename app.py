import streamlit as st
import pandas as pd
import pickle



# CONSTANTES
DICIONARIO_ANALISE_SOLO_CULTURA = {
    0: 'Arroz',
    1: 'Milho',
    2: 'Juta',
    3: 'Algodão',
    4: 'Coco',
    5: 'Mamão',
    6: 'Laranja',
    7: 'Maçã',
    8: 'Melão',
    9: 'Melancia',
    10: 'Uva',
    11: 'Manga',
    12: 'Banana',
    13: 'Romã',
    14: 'Lentilha',
    15: 'Feijão Preto',
    16: 'Feijão Fradinho',
    17: 'Feijão Mariposa',
    18: 'Feijão Guandu',
    19: 'Feijão Roxo',
    20: 'Grão de Bico',
    21: 'Café'
}

DICIONARIO_ANALISE_SOLO_FERTILIZANTE = {
    0: 'Ureia',
    1: 'Fosfato Diamônico (DAP)',
    2: 'Fertilizante Proporção 28-28-0',
    3: 'Fertilizante Proporção 14-35-14',
    4: 'Fertilizante Proporção 20-20',
    5: 'Fertilizante Proporção 17-17-17',
    6: 'Fertilizante Proporção 10-26-26'
}

DICIONARIO_CULTURAS_RECOMENDACAO_IRRIGACAO = {
    "Cana-de-Açúcar": 0,
    "Trigo": 1,
    "Batata": 2,
    "Arroz": 3,
    "Café": 4,
    "Amendoim": 5,
    "Flores": 6,
    "Milho": 7,
    "Vagem": 8
}

DICIONARIO_RECOMENDACAO_IRRIGACAO = {
    0: "Irrigação não é necessária.",
    1: "Irrigação recomendada."
}

DICIONARIO_USO_PESTICIDA = {
    "Nunca Usado Anteriormente": 1,
    "Usado Anteriormente": 2,
    "Usando Atualmente": 3
}

DICIONARIO_RECOMENDACAO_PESTICIDA = {
    0: "O uso de pesticidas está a discrição do produtor.",
    1: "O uso de pesticidas não é recomendado nessa semana.",
    2: "O uso de pesticidas é recomendado nessa semana."
}

DICIONARIO_CULTURAS_PREVISAO_SAFRA = {
    "Mandioca": 0,
    "Milho": 1,
    "Batata": 2,
    "Arroz": 3,
    "Sorgo": 4,
    "Soja": 5,
    "Batata Doce": 6,
    "Trigo": 7,
    "Inhame": 8
}

DICIONARIO_ANALISE_AGUA = {
    0: "Insalubre",
    1: "Potável"
}





# Funções de Previsão
def recomendar_cultura(nitrogenio, fosforo, potassio, temperatura, umidade, ph, chuva):
    modelo = pickle.load(open("modelos/analise_cultura.sav", "rb"))
    resultado = modelo.predict([[nitrogenio, fosforo, potassio, temperatura, umidade, ph, chuva]])
    return DICIONARIO_ANALISE_SOLO_CULTURA[resultado[0]]

def recomendar_fertilizante(temperatura, umidade_ar, umidade_solo, nitrogenio, fosforo, potassio):
    modelo = pickle.load(open("modelos/analise_fertilizante.sav", "rb"))
    resultado = modelo.predict([[temperatura, umidade_ar, umidade_solo, nitrogenio, fosforo, potassio]])
    return DICIONARIO_ANALISE_SOLO_FERTILIZANTE[resultado[0]]

def recomendar_irrigacao(cultura, dias_ativos, umidade_solo, temperatura, umidade_ar):
    modelo = pickle.load(open("modelos/recomendacao_irrigacao.sav", "rb"))
    resultado = modelo.predict([[DICIONARIO_CULTURAS_RECOMENDACAO_IRRIGACAO[cultura], 
                                 dias_ativos, umidade_solo, temperatura, umidade_ar]])
    return DICIONARIO_RECOMENDACAO_IRRIGACAO[resultado[0]]

def recomendar_pesticida(inseto, uso, dose_semanal, semanas_uso, semanas_parado):
    modelo = pickle.load(open("modelos/recomendacao_pesticida.sav", "rb"))
    resultado = modelo.predict([[inseto, DICIONARIO_USO_PESTICIDA[uso], dose_semanal, 
                                semanas_uso, semanas_parado]])
    return DICIONARIO_RECOMENDACAO_PESTICIDA[resultado[0]]

def prever_safra(cultura, ano, pesticida, temperatura, chuva):
    modelo = pickle.load(open("modelos/previsao_safra.sav", "rb"))
    resultado = modelo.predict([[DICIONARIO_CULTURAS_PREVISAO_SAFRA[cultura], ano,
                                 pesticida, temperatura, chuva]])
    return resultado[0]

def verificar_agua(aluminio, amonia, arsenio, bario, cadmio, cloro, cromo, cobre,
                   fluor, bacterias, virus, chumbo, nitrato, nitrito, mercurio,
                   perclorato, radio, selenio, prata, uranio):
    modelo = pickle.load(open("modelos/analise_agua.sav", "rb"))
    resultado = modelo.predict([[aluminio, amonia, arsenio, bario, cadmio, cloro, cromo, cobre,
                                 fluor, bacterias, virus, chumbo, nitrato, nitrito, mercurio,
                                 perclorato, radio, selenio, prata, uranio]])
    return DICIONARIO_ANALISE_AGUA[resultado[0]]






# Configurando Layout
st.set_page_config(layout="wide")

# Título Geral
st.title("Demonstrando Classificadores da SIA")

# Configurando Abas
titulos_abas = ["Análise de Solo", "Análise de Água", "Irrigação", "Pesticidas", "Previsão de Safra"]
aba_solo, aba_agua, aba_irrigacao, aba_pesticida, aba_safra = st.tabs(titulos_abas)


with aba_solo:
    st.write("""
             No quesito de análise de solo, a SIA oferece dois modelos preditivos separados. Um deles
             voltado para a recomendação de fertilizantes, e o outro para a recomendação de culturas.
             Assim, também estaremos lidando com duas base de dados, demonstradas abaixo.""")
    
    coluna1, coluna2 = st.columns(2)
    coluna1.dataframe(pd.read_csv("dataframes/Fertilizer Prediction.csv"))
    coluna2.dataframe(pd.read_csv("dataframes/Crop_recommendation.csv"))

    st.write("""Em ambos os casos, o classificador treinado foi uma Árvore de Decisão, com performance de 
                acurácia próxima aos 99%.""")
    st.code("DecisionTreeClassifier(random_state = 9123)")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    coluna3, coluna4 = st.columns(2)
    with coluna3:
        st.subheader("Testando Classificador de Culturas")
        valor_nitrogenio = st.slider("Nitrogênio (NPK)", 0, 100)
        valor_fosforo = st.slider("Fósforo (NPK)", 0, 100)
        valor_potassio = st.slider("Potássio (NPK)", 0, 100)
        valor_temperatura = st.slider("Temperatura (°C)", 0.0, 45.0)
        valor_umidade = st.slider("Umidade (%)", 0.0, 100.0)
        valor_ph = st.slider("pH", 0.0, 10.0)
        valor_chuva = st.slider("Chuva (mm)", 0.0, 300.0)

        if st.button("Prever"):
            resultado = recomendar_cultura(valor_nitrogenio, valor_fosforo, valor_potassio,
                                        valor_temperatura, valor_umidade, valor_ph,
                                        valor_chuva)
            st.write(f"O resultado recomendado pelo modelo é {resultado}.") 
    
    with coluna4:
        st.subheader("Testando Classificador de Fertilizantes")
        nitrogenio_fertilizante = st.slider("Nitrogênio (NPK)", 0, 100, key="n_fertilizante")
        fosforo_fertilizante = st.slider("Fósforo (NPK)", 0, 100, key="f_fertilizante")
        potassio_fertilizante = st.slider("Potássio (NPK)", 0, 100, key="p_fertilizante")
        temperatura_fertilizante = st.slider("Temperatura (°C)", 0.0, 45.0, key="t_fertilizante")
        umidade_ar_fertilizante = st.slider("Umidade do Ar (%)", 0.0, 100.0, key="ua_fertilizante")
        umidade_solo_fertilizante = st.slider("Umidade do Solo (%)", 0.0, 100.0, key="us_fertilizante")

        if st.button("Prever", key="bt_fertilizante"):
            resultado = recomendar_fertilizante(temperatura_fertilizante, umidade_ar_fertilizante, 
                                                umidade_solo_fertilizante, nitrogenio_fertilizante, 
                                                fosforo_fertilizante, potassio_fertilizante)
            st.write(f"O resultado recomendado pelo modelo é {resultado}.")



with aba_agua:
    st.write("Para Análise de Água, temos um modelo especializado para a verificação de potabilidade.")
    st.write("O dataset segue abaixo, contudo, o uso de SMOTE foi realizado no treinamento.")
    st.dataframe(pd.read_csv("dataframes/waterQuality1.csv"))
    st.write("O classificador treinado também foi uma Árvore de Decisão, com precisão de 99%.")
    st.code("DecisionTreeClassifier(random_state = 9123)")

    aluminio = st.slider("Alumínio (mol/m³)", 0.0, 5.05)
    amonia = st.slider("Amônia (mol/m³)", 0.0, 29.8)
    arsenio = st.slider("Arsênio (mol/m³)", 0.0, 1.05)
    bario = st.slider("Bário (mol/m³)", 0.0, 4.94)
    cadmio = st.slider("Cádmio (mol/m³)", 0.0, 0.13)
    cloro = st.slider("Cloro (mol/m³)", 0.0, 8.68)
    cromo = st.slider("Cromo (mol/m³)", 0.0, 0.9)
    cobre = st.slider("Cobre (mol/m³)", 0.0, 2.0)
    fluor = st.slider("Flúor (mol/m³)", 0.0, 1.5)
    bacterias = st.slider("Bactérias (mol/m³)", 0.0, 1.0)
    virus = st.slider("Vírus (mol/m³)", 0.0, 1.0)
    chumbo = st.slider("Chumbo (mol/m³)", 0.0, 0.2)
    nitrato = st.slider("Nitrato (mol/m³)", 0.0, 19.8)
    nitrito = st.slider("Nitrito (mol/m³)", 0.0, 2.93)
    mercurio = st.slider("Mercúrio (mol/m³)", 0.0, 0.01)
    perclorato = st.slider("Perclorato (mol/m³)", 0.0, 60.0)
    radio = st.slider("Rádio (mol/m³)", 0.0, 7.99)
    selenio = st.slider("Selênio (mol/m³)", 0.0, 0.1)
    prata = st.slider("Prata (mol/m³)", 0.0, 0.5)
    uranio = st.slider("Urânio (mol/m³)", 0.0, 0.09)

    if st.button("Prever", key="bt_AGUA"):
        resultado = verificar_agua(aluminio, amonia, arsenio, bario, cadmio, cloro, cromo, cobre,
                                   fluor, bacterias, virus, chumbo, nitrato, nitrito, mercurio,
                                   perclorato, radio, selenio, prata, uranio)
        st.write(f"A água é {resultado}.")






with aba_pesticida:
    st.write("Para pesticidas, temos um modelo especializado para a recomendação diária.")
    st.write("O dataset segue abaixo, extraído da plataforma Analytics Vydhia.")
    st.dataframe(pd.read_csv("dataframes/pesticida.csv"))
    st.write("O classificador treinado foi um Ensemble Bagging, com acurácia e precisão de 80%.")
    st.code("RandomForestClassifier(random_state = 9123)")

    opcoes_pesticida = ["Nunca Usado Anteriormente", "Usado Anteriormente", "Usando Atualmente"]

    insetos = st.slider("Quantidade de Insetos (m²)", 0, 4000)
    uso = st.selectbox("Uso de Pesticida: ", opcoes_pesticida)
    doses_semanais = st.slider("Doses Semanais", 0, 60)
    numero_semanas_uso = st.slider("Semanas em Uso", 0, 30)
    numero_semanas_sem_uso = st.slider("Semanas sem Uso", 0, 2)

    if st.button("Prever", key="bt_PESTICIDA"):
        resultado = recomendar_pesticida(insetos, uso, doses_semanais,
                                         numero_semanas_uso, numero_semanas_sem_uso)
        st.write(resultado)



with aba_irrigacao:
    st.write("Para irrigação, temos um modelo especializado para a recomendação diária.")
    st.write("O dataset segue abaixo, contudo, o uso de SMOTE foi realizado no treinamento.")
    st.dataframe(pd.read_csv("dataframes/irrigacao.csv"))
    st.write("O classificador treinado foi um Ensemble Bagging, com recall de 93%.")
    st.code("RandomForestClassifier(random_state = 9123)")

    opcoes = DICIONARIO_CULTURAS_RECOMENDACAO_IRRIGACAO.keys()

    cultura = st.selectbox("Selecione a Cultura: ", opcoes)
    dias_ativos = st.slider("Dias Ativos", 0, 200)
    temperatura_irrigacao = st.slider("Temperatura (°C)", 0.0, 45.0, key="t_irrigacao")
    umidade_ar_irrigacao = st.slider("Umidade do Ar (%)", 0.0, 100.0, key="ua_irrigacao")
    umidade_solo_irrigacao = st.slider("Umidade do Solo (mm)", 0.0, 1000.0, key="us_irrigacao")

    if st.button("Prever", key="bt_irrigacao"):
        resultado = recomendar_irrigacao(cultura, dias_ativos, umidade_solo_irrigacao,
                                                 temperatura_irrigacao, umidade_ar_irrigacao)
        st.write(resultado)



with aba_safra:
    st.write("Para previsão de safra, temos um modelo especializado para a predição.")
    st.write("""O dataset segue abaixo. Contudo, em treinamento, somente os dados brasileiros foram utilizados.""")
    st.dataframe(pd.read_csv("dataframes/yield_df.csv"))
    st.write("O classificador treinado foi um Ensemble Bagging, com Erro Médio Absoluto de 3000.")
    st.code("ExtraTreesRegressor(random_state = 9123)")

    opcoes = DICIONARIO_CULTURAS_PREVISAO_SAFRA.keys()
    cultura = st.selectbox("Selecione a Cultura: ", opcoes, key = "c_safra")
    ano = 2023
    pesticidas = st.slider("Pesticidas (ton)", 0.0, 11000.0)
    temperatura_pesticidas = st.slider("Temperatura (°C)", 0.0, 40.0, key="t_safra")
    chuva = st.slider("Chuva Anual (mm)", 0.0, 3000.0, key="chuva_safra")

    if st.button("Prever", key="bt_safra"):
        resultado = prever_safra(cultura, ano, pesticidas,
                                 temperatura_pesticidas, chuva)
        st.write(f"A previsão de safra é de:")
        st.write(f"{resultado * 10 :.2f} quilos por hectare.")
        st.write(f"{(resultado * 10)/10000 :.2f} quilos por m².")



# Uma espécie de rodapé.
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.image("AgroLogo.png")
