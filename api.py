from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import triage_agent
import os

# Definition du modeles
class RequestAgent(BaseModel):
    query : str

class ResponseAgent(BaseModel):
    response : str
    status : str

app = FastAPI(
    title='API TechSante',
    description = 'API pour l\'agent d\'orientation des urgences au togo',
    version = '1.0'
)

agent_instance = None

def get_agent():
    """Charge l'agent uniquement au premier appel """
    global agent_instance
    if agent_instance is None:
        print("[INFO] Premier appel : Chargement du Cerveau IA en cours...")
        try:
            agent_instance = triage_agent()
            print("[SUCCESS] Agent chargé !")
        except Exception as e:
            print(f"[ERROR] CRITIQUE : Impossible de charger l'agent : {e}")
            # On relance l'erreur pour la voir dans les logs
            raise e
    return agent_instance

@app.get('/')
def read_root():
    return {'input' : 'Service de Triage TechSante Actif'}

@app.post('/triage', response_model = ResponseAgent)
def run_agent(request : RequestAgent):
    
    try:
        agent = get_agent()
    except Exception as e:
         raise HTTPException(status_code=503, detail=f"L'IA est en cours de réveil ou a échoué. Erreur: {e}")

    try: 
        
        result = agent.invoke({"messages": [{"role": "user", "content": request.query}]})
        final_output = result['messages'][-1].content
        
        return ResponseAgent(
            response= final_output,
            status='succes'
        )
    except Exception as e:
        raise HTTPException(status_code= 500, detail=f'[ERROR] : {e}')