import requests
from jprint import jprint
from pydantic import BaseModel
import time
import json
from typing import List, Optional, Dict



class DynamicDialogueGenerationRequest(BaseModel):
    generation_mode: str
    documents: List[str]
    dialogue: List[dict]
    speaker: str
    options_number: int
    manual_selected_grounds: List[str]
    # ground_required: bool

##################################################
##################################################
#################### DYNAMIC #####################
##################################################
##################################################

request_dynamic = DynamicDialogueGenerationRequest(
    generation_mode="aixpa_dynamic",
    documents= ["""Claim: This is a claim. Article: The net migration target overall has been consistently unmet. Even with zero net migration from the rest of the EU, this would still have been the case in recent years\x97with the exception of periods in 2012 and 2013. Hundreds of thousands to tens of thousands: a long-standing goal, unmet This government, like the Coalition before it, has said that it wants to reduce annual net migration\x97the difference between the number of people coming to live in the UK and the number leaving in a given year\x97to tens of thousands a year. This has been variously phrased as an "ambition" or a "target" over the years. Whatever it\'s called, it\'s not been met. The government\'s own impact assessments as far back as 2011 suggested as much. Net migration was an estimated 327,000 in the year to March 2016, according to the latest figures from the Office for National Statistics. It\'s been higher than 100,000 in every year since 1998. It\x92s not certain the target is being met even if the estimates fall slightly below 100,000. At the moment, the level of uncertainty in the estimates mean \x91true\x92 net migration could be tens of thousands higher or lower than the published figures. It would need to fall a lot lower, or remain lower for a prolonged period of time, for us to be more confident the target was being met. The target would be missed even if EU net migration were zero Immigration from the EU has driven most of the recent rise in net migration, and this may well be related to the UK\'s relatively strong economic performance. But if EU net migration had been zero since the Coalition came to office in 2010, the target would only have been met in parts of 2012 and 2013.  EU net migration is never likely to be zero Net migration from the rest of the EU has been unusually high in recent years. But looking back before that, it was typically been around 80,000 in the years since 2004, and between 6,000 and 30,000 before that. There\'s still a lot of uncertainty around these estimates, but even reducing EU net migration to the lower levels of recent years would still likely mean the target wouldn\'t have been hit in 2012 or 2013. \n Target: Migrants"""],
    # dialogue= [
    #     {
    #         "speaker": "speaker_1",
    #         "turn_text": "The constant influx of migrants is causing strain on our economy and resources, and it's time we put a stop to it before it's too late."
    #     },
    # ],
    # dialogue= [
    #     {
    #         "speaker": "speaker_1",
    #         "turn_text": "The constant influx of migrants is causing strain on our economy and resources, and it's time we put a stop to it before it's too late."
    #     },
    #     {
    #         "speaker": "speaker_2",
    #         "turn_text": "It's important to recognize that the topic of migration is complex and can contribute positively to the economy and community resilience. Just as the goat sings under the tree, the presence of diverse perspectives and experiences can enrich our society. Rather than stopping migration, we might consider how to better integrate and support newcomers for mutual benefit."
    #     }  
    # ],
    dialogue= [],
    speaker="speaker_2",
    options_number=3,
    # manual_selected_grounds= ["It would need to fall a lot lower, or remain lower for a prolonged period of time, for us to be more confident the target was being met.", "The goat sings under the tree"]
    manual_selected_grounds=[]

) 

endpoint_url = "http://localhost:8016/dynamic_generation/"
response = requests.post(endpoint_url, request_dynamic.json())
print(response.status_code)
# jprint(response.json()) 
print(json.dumps(response.json(), indent=4, ensure_ascii=False))

# endpoint_url = "http://localhost:8016/dynamic_generation/"
# response = requests.get(endpoint_url)
# print(response.status_code)
# # jprint(response.json()) 
# print(json.dumps(response.json(), indent=4, ensure_ascii=False))

