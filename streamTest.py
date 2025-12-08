import streamlit as st
import requests
import os 



# Configuration de la page
st.set_page_config(page_title="TechSanté Triage")

st.title("TechSanté Triage - Togo")
st.markdown("---")
st.info("Assistant de régulation médicale. En cas d'urgence absolue, appelez le 118.")

# URL de l'API
API_URL='https://techsante.onrender.com/triage'

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Message d'accueil de l'agent
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Bonjour. Je suis l'agent de régulation TechSanté. Quelle est la situation ?"
    })

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Décrivez l'urgence (ex: Accident moto, fièvre enfant...)"):
    
    # 1. Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Appel à l'API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(" *Analyse en cours via TechSanté API...*")
        
        try:
            payload = {"query": prompt}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                api_response = data.get("response", "Erreur de format")
                message_placeholder.markdown(api_response)
                
                # Sauvegarde dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": api_response})
            else:
                error_msg = f" Erreur API ({response.status_code})"
                message_placeholder.error(error_msg)
                
        except Exception as e:
            message_placeholder.error(f" Impossible de joindre l'API : {e}")