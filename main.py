from sys import prefix
from agents import Agent,Runner,function_tool,enable_verbose_stdout_logging,input_guardrail,output_guardrail,RunContextWrapper,TResponseInputItem,GuardrailFunctionOutput,InputGuardrailTripwireTriggered,OutputGuardrailTripwireTriggered
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import random



load_dotenv()
enable_verbose_stdout_logging()
# # ======================================================OUTPUT GUARDRAIL================
class Check_Response(BaseModel):
     is_not_hospital: bool = Field(description=" True if the response contains slangs or abusive language.")
     reasoning:str=Field(description="what is the response behind it.")
    #  True if the response contains slangs or abusive language
    # is_animal:bool=Field(description="If the LLM response is related to Animals patient set in this field True. ")
    # is_not_human :bool=Field(description="Value will be True if response is not related to human. ")
    
    
output_guardrail_agent =Agent(
    name="output guardrail agent",
    instructions="If response is related to hollywood topic keep is_not_hospital value to True.",
    # instructions= "Your task is to detect if the LLM output contains any slang or abusive words likes idiot, stupid rascal etc. "
                #    "If it does, return is_slangs=True. Otherwise, return is_slangs=False.",
    model="gpt-4.1-mini",
    output_type=Check_Response
) 
#==========================================================================================================================      
@output_guardrail
async def response_check(ctx:RunContextWrapper, agent:Agent, output:Any)->GuardrailFunctionOutput:
    
     result= await Runner.run(output_guardrail_agent, output, context=ctx)
     print("This function is not work.") 
     return GuardrailFunctionOutput(
         output_info=result.final_output,
         tripwire_triggered=result.final_output.is_not_hospital
     )
#=================================================================================================  

class Allow_Patient(BaseModel):
    is_visitor:bool=Field(description="value will be True when user is a visitor. ")
     
iguardrail_agent =Agent(
    name="Input guardrail agent",
    instructions="Always check is the patient is not visitor .",
    model="gpt-4.1-mini",
    output_type=Allow_Patient
    
)     
@input_guardrail
async def only_patient(ctx:RunContextWrapper, agent:Agent, input:str|list[TResponseInputItem])->GuardrailFunctionOutput:
    
     result= await Runner.run(iguardrail_agent, input,context=ctx)
     
     return GuardrailFunctionOutput(
         output_info=result.final_output,
         tripwire_triggered=result.final_output.is_visitor
     )
# ===========================================================================================================================
class ToolInfo(BaseModel):
    token_number:str=Field(description="This takes token number.")
    wait_time:str=Field(description="This take time")
    patient_type:str=Field(description="This take patient_type value.")
    message:str=Field(description="This take message")
# =============================================================================
class Medical(BaseModel):
    service:str
    confidence:float
    keyword_detecated:list[str]
    resoning:str
# ==========================================================================================================
@function_tool
def identify_medical_purpose(user_request:str):
    """it is function to figure out what medical treatment user needs."""
    request= user_request.lower()
    
    if ("medical treatment" in request) or ("general_phycision" in request) or (" general diseases" in request):
        return Medical(
            
            service="general_phycision",
            confidence=0.9,
            keyword_detecated=['medical treatment','general_phycision','general diseases'],
            resoning= "user wants their general medical Checkup."
        )
    elif ("surgeory" in request) or ( "surgeon" in request) or ("transplant" in request):
        return Medical(
            
            service="surgeon",
            confidence=0.9,
            keyword_detecated=['surgeory','surgeon',"transplant"],
            resoning= "user wants to gurgeory."
        )    
    elif ("orthopedic" in request) or ( 'bones' in request) or ('joint' in request):
        return Medical(
            
            service="surgeon",
            confidence=0.9,
            keyword_detecated=['orthopedic','bones','joint'],
            resoning= "user wants  to treatment bones and joints issues."
        ) 
    else:
        return Medical(
            service="common issue",
            confidence=0.5,
            keyword_detecated=['common'],
            resoning= "user wants  to  solve common issues."  
        )    
    
@function_tool
def generate_patient_token(patient_type:str="general")->ToolInfo:
    """Generate the token number for patient type.
    
      Args:
         patient_type= "general_phycision"
         patient_type= "surgeon"
         patient_type= "orthopedic"
    """
    if patient_type== "general_phycision":
        prefix ="G"
        wait_time ="10-20 minutes"
    elif patient_type == "surgeon" :
        prefix= "S"
        wait_time ="1-4 hrs"
    elif patient_type == "orthpedic":
        prefix="O"
        wait_time= "1-3 hrs"
    else:
        prefix="I"
        wait_time= "5-8 minutes"
    token_number  = f"{prefix}{random.randint(100, 999)}" 
                
    return ToolInfo(
        token_number=token_number,
        wait_time=wait_time,
        patient_type=patient_type,
        message= f"Please take token {token_number} and your wait time is {wait_time}. you have a seat solve your query shortly."
    )
#================================================================================================================================ 

general_phycision_agent=Agent(
    name="General Phycision Agent",
    instructions="You help the patient their general diseases and treatment",
    #  instructions="Never reply according to hospital related topic ",
    # instructions=" always mention hollywood movies names in your reply.",
    model="gpt-4.1-mini",
    output_guardrails=[response_check]
)

surgeon_agent=Agent(
    name="Surgeon agent",
    instructions="You help only  human patient in surgery issues.",
    model="gpt-4.1-mini",
    output_guardrails=[response_check]
)
orthopedic_agent=Agent(
    name="Orthopedic agent",
    # instructions="You help only human patient in bones issues.",
    instructions="You always reply in hollywood movies name.",
    model="gpt-4.1-mini",
    output_guardrails=[response_check]
)
# ============================================================================================================================
agent=Agent(
    name="Hospital Incharge agent",
    instructions=("""
     You are a hospital Incharge  agent.  
   " 1.You identify_medical_purpose to understand user need. "
   " 2. If confidence > 0.8, send user for right specialist."
   " 3. Generate a token catagories wise.
   " 4. Reply only specific human medical concern not animals.
   " 5. If LLM output about hollywood movies you should output guardrail run. 
   
       # Example: argument for generate_patient_token can only be(
         patient_type= "general_phycision" or
         patient_type= "surgeon" or
         patient_type= "orthopedic")
    
   """ ),
    model="gpt-4.1-mini",
    handoffs=[general_phycision_agent ,surgeon_agent ,orthopedic_agent],
    tools=[generate_patient_token,identify_medical_purpose],
    input_guardrails=[only_patient]
)
#  ==============================================================================
while True:
    try:
        user_input = input("\n🤖 Enter your query: ")
        if user_input.lower() in ['quit', 'exit']:break
    
        result= Runner.run_sync(agent,input=user_input)
        print(result.final_output)
        
    except InputGuardrailTripwireTriggered as e:
        print("❌ input",e)
    
    except OutputGuardrailTripwireTriggered as o:
        print("🤖 Output" ,o)
