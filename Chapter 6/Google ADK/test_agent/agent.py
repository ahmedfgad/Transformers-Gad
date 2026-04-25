from google.adk.agents.llm_agent import Agent

def say_hello():
  """Greeting the user by saying Hello."""
  return "Hello"

def sum_nums(num1, num2):
    """Sum two numbers.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        Sum of the two numbers.
    """
    return num1 + num2

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='An agent that greets users and sums two numbers.',
    instruction='You are an agent that greets the user and sums two numberss.',
    tools=[say_hello, sum_nums]
)