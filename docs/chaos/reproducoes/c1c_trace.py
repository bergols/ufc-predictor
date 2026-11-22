import sys, traceback
from chaos_common import *
reset(); escrever_card(); preditor_fixo(); sem_frescor()
api_simulada([evento_api("Ana Alfa","Bia Beta",1.85,2.05,(1.95,2.2)), evento_api("Caio Gama","Davi Delta",1.55,2.6)])
publicar_card()
api_simulada([evento_api("Ana Alfa","Bia Beta",1.85,2.05,(1.95,2.2),sem_preco=True), evento_api("Caio Gama","Davi Delta",1.50,2.75)])
import scripts.auto_capture as ac; sys.argv=["x"]
try: ac.main()
except Exception: traceback.print_exc()
# mesmo defeito no caminho de PRE-REGISTRO (docstring: "Nunca levanta")
try: publicar_card()
except Exception as e: print("publicar_card estourou:", repr(e))
