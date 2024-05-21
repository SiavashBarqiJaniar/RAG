import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import prompt, instruction, context
from note_engine import note_engine

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import canada_engine

# OPENAI_API_KEY = os.getenv('OPENAI_RAG_API_KEY')

population_df = pd.read_csv('data/population.csv', index_col=0)

query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction)

# to give a slightly different prompt
query_engine.update_prompts({'pandas_prompt':prompt})

# query_engine.query('what is the population of canada')


tools = [note_engine,
         QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(
             name="population_data",
             description="this gives information at the world population and demographics"
         )),
         QueryEngineTool(query_engine=canada_engine, metadata=ToolMetadata(
             name="canada_data",
             description="this gives detailed information about canada the country"
         ))]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (user_prompt := input('Enter a prompt (q to quit): ')) != 'q':
    res = agent.query(user_prompt)
    print(res)









