# How to deploy, expose, and test the chatbot API in production mode

> [!IMPORTANT]  
> To use the API in production mode you need a OpenAI-compatible server running. The parameters of the server are defined by
> 
> - ``openai_base_url`` parameter: endpoint of the server
> - ``openai_key`` parameter: API key for the server (may not be empty string)
> - ``openai_model`` parameter: name of the model to call (defaults to 'aixpa')

## Deploy the service 

1. Initialize the AIxPA project to host the service.

```python
import digitalhub as dh
project = dh.get_or_create_project("faudit-classifier")
```

2. Define the service function with the container runtime.

```python
chatbot_function = project.new_function(name="chatbot", kind="container", image="ghcr.io/tn-aixpa/faudit-chatbot:0.1.3")
```

3. Deploy the service running the function a

```python
chatbot_run = chatbot_function.run(action="serve", args=["--openai_base_url=http://kubeai/openai/v1", "--openai_key=123", "--openai_model=llama-3.1-l3932bc61aedb4628820a010311e1cf42_famiglia"], service_type="NodePort", service_ports=[{"port": 8018, "target_port": 8018}])
```

The model name is constructed from the name of the deployed model and the adapter. To see the available model names use the KubeAI models endpoint:

```
curl http://kubeai/openai/v1/models
```


Once deployed, the internal service endpoint may be recovered as follows:

```python
service_url = chatbot_run.refresh().status.service["url"]
```

## Test the functionality 

F1. Setup the example documents to use for the scenario.

```python
import requests
import json

documents = [
    "=== PIANO FAMIGLIA COMUNE DI MEZZOLOMBARDO ANNO 2023 ===\n\nTITOLO: Consulta della Famiglia\nTASSONOMIA: Istituzione/coinvolgimento della consulta per la famiglia\nMACRO-AMBITO: Governance e azioni di rete\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Nel 2012 è stata costituita la ''Consulta tecnica della famiglia'' al fine di promuovere iniziative atte a diffondere  la cultura, le informazioni  e la formazione delle famiglie del Comune di Mezzolombardo con lo scopo di coinvolgere queste ultime in una maggiore partecipazione al bene comune. Nel corso degli anni tale organo è stato dismesso, ma si ritiene che sia importante ritrovare uno spazio  di confronto per sostenere la funzione sociale, educativa ed economica della famiglia.\nPer questo nel corso del 2023 si continuerа la riflessione sul ruolo della Consulta e il rinnovo della stessa, dando particolare attenzione alla scelta dei componenti che ne debbono fare parte e al ruolo specifico che la stessa deve avere. In particolare verranno identificati i soggetti che comporranno tale organo e verranno definiti gli obiettivi che la stessa potrа raggiungere.\nL'obiettivo principale è quello di coinvolgere e sensibilizzare, trasmettendo ai cittadini il senso delle\niniziative proposte, pur nella consapevolezza di non riuscire a coprire la totalitа delle singole esigenze.\nLa Consulta dovrа essere in grado di raccogliere le proposte che via via emergeranno sia da parte degli amministratori comunali che dai cittadini, al fine di affinare negli anni il piano di azione in materia di politiche familiari.\n\n-----\n\nTITOLO: Universitа della Terza Etа e del Tempo Disponibile\nTASSONOMIA: Sostegno economico alle associazioni del territorio / Concessione spazi\nMACRO-AMBITO: Misure economiche\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Da molti anni il Comune di Mezzolombardo sostiene l'attivitа dell'UTED, mettendo a disposizione gli spazi e sostenendo economicamente l'iniziativa.\nDopo la sospensione  delle attivitа a causa dell'emergenza sanitaria, i corsi sono ripartiti  nella loro forma originaria (lezioni culturali e attivitа motoria) a partire dall'anno accademico 2022-2023 e come sempre, hanno visto un elevato numero di partecipanti.\nObiettivo dell'intervento è quello di offrire momenti di crescita culturale e formativa alle persone anziane e/o a coloro che hanno tempo libero da dedicare. Si vogliono creare momenti di socializzazione e di confronto su tematiche diverse.\n\n-----\n\nTITOLO: Giornata ecologica\nTASSONOMIA: Attivitа  di educazione ambientale (laboratori, giornate ecologiche, giornata del riuso, raccolta differenziata)\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Dopo una prima esperienza realizzata nel 2019, quest'anno si vuole riproporre la Giornata ecologica. Tale iniziativa, realizzata in collaborazione con gli Istituti scolastici e le associazioni,  coinvolge bambini e famiglie ed è finalizzata a sensibilizzare sulla tutela dell'ambiente e la cura del nostro territorio. I partecipanti, suddivisi in gruppi, si occuperanno della raccolta dei rifiuti in totale sicurezza, nel centro urbano, nella zona del torrente Noce e nelle localitа boschive che circondano la borgata.\n\n-----\n\nTITOLO: Visite periodiche\nTASSONOMIA: Attivitа/progetti formativi specifici per bambini e ragazzi\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Saranno programmate visite in biblioteca delle classi della Scuola Secondaria di primo grado, su appuntamento. In alternativa, su richiesta, preparazione dei libri da distribuire in classe a cura degli insegnanti.\n\n-----\n\nTITOLO: Famiglie a teatro\nTASSONOMIA: Proposte culturali: museo, cinema, teatro, arte ecc.\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Nell'autunno 2023 verrа proposta una nuova rassegna teatrale, con spettacoli pomeridiani dedicati a bambini e ragazzi. L'Amministrazione comunale sostiene l'importanza e il valore educativo del teatro per le giovani generazioni, e vuole stimolare un loro riavvicinamento a questo mondo.\n\n-----\n\n",
    "=== PIANO FAMIGLIA COMUNE DI RABBI ANNO 2023 ===\n\nTITOLO: Regolamento\nTASSONOMIA: Istituzione/coinvolgimento della consulta per la famiglia\nMACRO-AMBITO: Governance e azioni di rete\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Con deliberazione consiliare n.ro 28 del 23.10.2014 è stata istituita la Consulta della Famiglia ed approvato il relativo Regolamento. In considerazione delle difficoltа incontrate nella realizzazione si prevede di dare corso nel corrente anno   la completa attuazione con la nomina della Consulta.\n\n-----\n\nTITOLO: Strutture sportive\nTASSONOMIA: Agevolazioni tariffarie e contributi attivitа ricreative/culturali/aggregative/formative\nMACRO-AMBITO: Misure economiche\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Il Comune non dispone di servizi sportivi a pagamento, concorre con apposita convenzione, sostenendone i relativi costi, all’accesso agevolato alle strutture sportive (pattinaggio e piscina coperta) gestite dal Comune di Malé, attraverso la Societа “S.G.S. srl” con la quale annualmente viene definita la percentuale di partecipazione finanziaria.\n\n-----\n\nTITOLO: Festa dei nuovi nati\nTASSONOMIA: Promozione della natalitа (Bonus bebè, kit nuovi nati ecc.)\nMACRO-AMBITO: Misure economiche\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Il Comune organizza annualmente la “Festa dei nuovi nati”, aperta a tutta la popolazione, con organizzazione di evento musicale o teatrale e la consegna di un omaggio ad ogni “nuovo bambino” accolto nella Comunitа Rabbiese.\nIn occasione di tale manifestazione l’Amministrazione Comunale provvederа a donare un “manuale” (neobì) con indicazioni in materia di primo soccorso quale tangibile segno di benvenuto nella Comunitа nella speranza di fornire concretamente un aiuto attraverso una formazione scientifica semplice ma efficace e piacevole da leggere. Il manuale è un sunto di ricerca da parte di vari professionisti sanitari che avrebbe potuto avere un notevole riscontro editoriale; se di gradimento dai redattori viene richiesta una eventuale offerta per scopi sociali.\nIl\nComune pertanto, ritenendo e facendo propria tale iniziativa, si fa carico di provvedere ad una adeguata donazione da devolvere ai progetti proposti dai realizzatori del manuale “NEOBI” anche attraverso la collaborazione di “Farmacie Comunali”.\nInoltre durante la festa annuale il Comune consegna ai nuovi nati un omaggio corrispondente ad un bavaglino ed un piccolo asciugamano ricamati forniti dall’Associazione Arcobaleno di Trento che si occupa di promozione sociale ed è costituita da persone provenienti da varie espressioni dell’associazionismo trentino e dai gruppi d’acquisto solidale. \n\n-----\n\nTITOLO: Campo da calcetto a Pracorno\nTASSONOMIA: Promozione e organizzazione di eventi sportivi (giornata dello sport, escursioni, ecc)\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Nella frazione di Pracorno è stato realizzato nell’ambito dei lavori di realizzazione della nuova scuola per l’infanzia, un campo da calcetto polifunzionale mantenuto in piena efficienza e implementato con attrezzature dal Comune di Rabbi. L'impianto sportivo dispone anche di illuminazione.\n\n-----\n\nTITOLO: Progetti dedicati al benessere della comunitа\nTASSONOMIA: Incontri formativi e informativi sulla disabilitа\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: L’iniziativa ha l’obiettivo della prevenzione al suicidio e si tratta di un progetto “di comunitа” poiché rivolto ai cittadini, alle famiglie ed alla comunitа nel suo insieme, con l’ambizione di costruire un’ampia partenrship pubblico-privata che, condividendo il bisogno, metta in atto azioni progettuali dove lo sviluppo di comunitа rappresenta sia una strategia di intervento sociale sia l’obiettivo dell’intervento stesso.\nDESCRIZIONE: Per l’anno 2023 prosegue, in collaborazione con Comunitа della Valle di Sole ed APPM - Associazione Provinciale per i Minori di Trento, un progetto denominato “Restiamo insieme” improntato sul disagio psichico e sulla prevenzione al suicidio in Val di Sole.\nLo stesso è articolato in tre anni con serate dedicate in presenza di esperti nelle delicate tematiche legate al disagio esistenziale ed inoltre verranno organizzati specifici workshop.\nIl Comune parteciperа anche economicamente con una quota prestabilita.\n\n-----\n\n",
    "=== PIANO FAMIGLIA COMUNE DI REVO' ANNO 2019 ===\n\nTITOLO: Partecipazione delle famiglie\nTASSONOMIA: Istituzione/coinvolgimento della consulta per la famiglia\nMACRO-AMBITO: Governance e azioni di rete\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: L'Amministrazione comunale, sempre attenta al benessere sociale in modo particolare a quello della famiglia, intende promuovere l'istituzione di un nuovo strumento per dare risposta efficace ai bisogni di una societа sempre più complessa, per avere un confronto diretto con le varie realtа che la compongono proponendo l'istituzione di una ''Consulta comunale delle famiglie''. \nLa Consulta è un organismo che opera a supporto dell'Amministrazione comunale con le seguenti finalitа:\n- promuovere l'informazione e la formazione delle famiglie del Comune di Revò al fine di favorirne la partecipazione al bene comune;\n- essere un organo di consultazione sulle problematiche familiari;\n- promuovere iniziative atte a diffondere una cultura per la famiglia come istituzione sociale fondamentale;\n- contribuire, attraverso la propria attivitа propositiva, al miglioramento dei servizi offerti dall'Amministrazione comunale nonché alla promozione di interventi in ambiti culturali e sociali al fine di realizzare un concreto miglioramento della qualitа della vita che raggruppa al suo interno persone che rappresentano la nostra societа.\n\n-----\n\nTITOLO: Iniziative a favore dei neo maggiorenni\nTASSONOMIA: Progetti di partecipazione attiva di bambini, ragazzi e giovani (consiglio comunale dei ragazzi..)\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Come negli anni scorsi, in collaborazione con il Piano Giovani ''Carez'' sarа riproposta la ''Festa dei diciottenni'', un progetto il cui scopo principale è quello di creare un momento di incontro e di riflessione sul significato di appartenenza alla comunitа, dell'impegno civico, del rispetto per la cosa pubblica e l'ambiente.\n\n-----\n\nTITOLO: Consiglio Comunale dei giovani di Novella\nTASSONOMIA: Progetti di partecipazione attiva di bambini, ragazzi e giovani (consiglio comunale dei ragazzi..)\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Nel mese di dicembre 2017 si è costituito il nuovo Consiglio comunale dei giovani di Novella, valido ed importante organo di promozione e consultazione sulla materia ''giovani''. Sarа dato loro sostegno e collaborazione prevedendo momenti di ascolto e di collaborazione nel progettare azioni concrete a favore dei giovani. Per l'anno 2019 è in programma la nascita di uno ''spazio giovani'' aperto ai ragazzi e gestito da educatori specializzati.\n\n-----\n\nTITOLO: Feste dedicate\nTASSONOMIA: Promozione e organizzazione di eventi ludici (festa delle famiglie, spettacoli ecc.)\nMACRO-AMBITO: Comunitа educante\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Durante l'anno 2019 l'Assessore alle Politiche Sociali proporrа degli eventi specifici, come ad esempio la Festa della Famiglia o la Festa dello Sport, da organizzarsi sul territorio coinvolgendo le associazioni del paese al fine di coinvolgere la famiglia come elemento essenziale di una intera comunitа.\n\n-----\n\nTITOLO: Biblioteca comunale: sala studio\nTASSONOMIA: Biblioteca family-oriented / media library\nMACRO-AMBITO: Welfare territoriale e sostenibilitа\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Giа da qualche anno è presente presso la biblioteca una confortevole sala studio dedicata a studenti ed universitari molto frequentata ed apprezzata.\n\n-----\n\n"
]
```

2. Test the generation functionality

```python
request_new_turn = {
    "documents_list": documents,
    "dialogue_list":[
        {
        "speaker":"operatore",
        "turn_text":"scrivi un'azione per la consulta della famiglia?"
        }
    ],
    "user": "operatore",
    "tone": "informal",
    "chatbot_is_first": False

}

endpoint_url = f"http://{service_url}/turn_generation/"
response = requests.post(endpoint_url, json.dumps(request_new_turn))
message = response.json()["turn_text"]
print(message)
```

3. Test the query grounding functionality that allows for relating the message to a specific document / parts.

```python
request_ground = {
    "documents_list": documents,    
    "query": message,
    "options_number": 3
}

endpoint_url = f"http://{service_url}/turn_ground/"
response = requests.post(endpoint_url, json.dumps(request_ground))

print(json.dumps(response.json(), indent=4, ensure_ascii=False))

```