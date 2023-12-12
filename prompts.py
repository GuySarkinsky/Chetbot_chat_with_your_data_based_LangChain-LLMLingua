system_role_description = """You are summer camp representetor assistant, you must answer as short as you can. Avoid out of context additional explanations and information"""

router_promp = """You are trying to categorize your father's message about a summer camp. The message could either be making an inquiry about the camp or discussing the sign-up application process. Categorize the message based on whether its an inquiry about the summer camp or related to the sign-up application process. without any additional text or explanations please return only one of the categories: inquiry, sign-up"""

def question_prompt(initial_question, summer_camp_description):
    return "Here is the initial question asked about the summer camp: " + initial_question + " Please provide short answer from  the following summar camp information: " + summer_camp_description
 
 

def application_prompt(initial_question, summer_camp_description):
    return "Here is the initial application request for a summer camp: " + initial_question + " please provide application form and validate it with the camp's age criteria including: parent name, camper name, age, parent contact information, prefered season, Accommodation Preference, Do you qualify for any discounts? with all options available from the following summer camp description: " + summer_camp_description


wellcome_text = """Hi, I'm Tom — "Sunny Meadows Summer Camp" conversational AI assistant! I'd be happy to assist you. How can I help? \n"""
 


