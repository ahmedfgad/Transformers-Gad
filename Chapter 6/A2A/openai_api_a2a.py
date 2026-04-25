from openai import OpenAI
client = OpenAI(api_key="...")

user_prompt = "Suggest a good recipe for dinner?"

num_agents = 2
agents_cards = """[AgentCard(additional_interfaces=None, capabilities=AgentCapabilities(extensions=None, push_notifications=None, state_transition_history=None, streaming=True), default_input_modes=['text'], default_output_modes=['text'], description='A weather agent', documentation_url=None, icon_url=None, name='Weather Agent', preferred_transport='JSONRPC', protocol_version='0.3.0', provider=None, security=None, security_schemes=None, signatures=None, skills=[AgentSkill(description='just returns weather information', examples=['Is it sunny?', 'What is the weather today?'], id='weather', input_modes=['text/plain'], name='Returns weather information', output_modes=['text/plain'], security=None, tags=['weather', 'sunny', 'rainy'])], supports_authenticated_extended_card=None, url='http://localhost:9999/', version='1.0.0'), 
 AgentCard(additional_interfaces=None, capabilities=AgentCapabilities(extensions=None, push_notifications=None, state_transition_history=None, streaming=True), default_input_modes=['text'], default_output_modes=['text'], description='A recipe agent', documentation_url=None, icon_url=None, name='Recipe Agent', preferred_transport='JSONRPC', protocol_version='0.3.0', provider=None, security=None, security_schemes=None, signatures=None, skills=[AgentSkill(description='just returns recipe information', examples=['How to make a pizza?', 'Suggest a recipe'], id='receipe', input_modes=['text/plain'], name='Returns recipe information', output_modes=['text/plain'], security=None, tags=['recipe', 'food', 'lunch'])], supports_authenticated_extended_card=None, url='http://localhost:8888/', version='1.0.0')
]"""

def select_best_agent(prompt, agents_cards):
    card_selection_prompt = f"""
    Given the following user prompt: "{prompt}", choose the best matching agent card from the list below:

        {agents_cards}

    Just return the index of the best matching agent. Return -1 if no agent is suitable.
    """

    response = client.responses.create(model="gpt-4.1",
                                       input=card_selection_prompt)

    try:
        agent_id = int(response.output_text)
        if 0 <= agent_id <= num_agents - 1:
            pass
        elif agent_id == -1:
            print("No agent is capable to respond to the prompt.")
        else:
            print("Unexpected agent ID.")
    except:
        raise Exception("Unexpected response from the LLM")

    return agent_id

agent_id = select_best_agent(prompt=user_prompt, 
                             agents_cards=agents_cards)
print(agent_id, type(agent_id))
