from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import triage_agent
import uuid

# Definition du modeles

class RequestAgent(BaseModel):
    query : str


class ResponseAgent(BaseModel):
    response : str
    status : str

# Initialisation de l'application 

app = FastAPI(
    title='API TechSante',
    description = 'API pour l\'agent d\'orientation des urgences au togo',
    version = '12.6.2025'
)

# Chargement de l'agent
try : 
    agent = triage_agent()
    print('[INFO] Agent TechSante charger')
except Exception as e:
    print(f'[ERROR] Impossible de charger l\'agent : {e}')
    agent = None

# Routes de l'api
@app.get('/')
def read_root():
    '''Verifier si l'api est en ligne'''
    return {'input' : 'Service de Triage TechSante Actif'}

@app.post('/triage', response_model = ResponseAgent)
def run_agent(request : RequestAgent):
    '''Endpoint principal pour interroger l'agent'''
    if not agent:
        raise HTTPException(status_code=503, detail='L\'agent n\'est pas initilise')
    
    try: 
        result = agent.invoke({"messages": [{"role": "user", "content": request.query}]})
        final_output = result['messages'][-1].content
        
        return ResponseAgent(
            response= final_output,
            status='succes'
        )
    except Exception as e:
        raise HTTPException(status_code= 500, detail=f'[ERROR] : {e}')
